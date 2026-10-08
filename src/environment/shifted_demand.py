"""Joint seven-day demand and traffic assignment with verified continuation.

The exact inverse-demand integral is convex on z=q-a>0. For numerical trial
points ONLY, extend it below a small positive threshold with its tangent
quadratic. This permits the old bound-constrained path optimizer to recover
from an infeasible trial step. A solution is NEVER accepted in this extension:
all routed ODs must satisfy z>threshold and the original fixed point and UE gap.
If necessary the threshold shrinks. Thus accepted points solve the unchanged
convex shifted-demand problem, not a smoothed behavior model.
"""
import numpy as np
from scipy.optimize import minimize
from src.environment.ue import _Network
from src.environment.evaluate import _matrix_from_H
from src.environment.fixed_point_demand import _shortest_paths, _incidence_matrix


def potential_factory(network, incidence, path_od, a, b, reference, gamma, reachable, floor):
    def objective(f):
        x = np.asarray(incidence@f).ravel()
        q = np.bincount(path_od, weights=f, minlength=len(a))
        z = q-a
        safe = np.maximum(z, floor)
        log_ratio = np.log(safe/b)
        entropy = (safe*log_ratio-safe)/gamma-reference*q
        below = z < floor
        delta = z[below]-floor[below]
        entropy[below] += np.log(floor[below]/b[below])*delta/gamma + delta**2/(2*gamma*floor[below])
        marginal = log_ratio/gamma-reference
        marginal[below] += delta/(gamma*floor[below])
        potential = np.sum(network.t0*(x+network.alpha*x**(network.beta+1)
                          /((network.beta+1)*network.cap**network.beta)))+entropy[reachable].sum()
        gradient = np.asarray(incidence.T@network.cost(x)).ravel()+marginal[path_od]
        return float(potential), gradient
    return objective


def solve_shifted(ctx, edges, external, reference, gamma, a, b, tolerance=.01,
                  warm_start=None, max_rounds=40, ue_tolerance=1e-3):
    external, reference, a, b = [np.asarray(v,float) for v in (external,reference,a,b)]
    if a.min() < -1e-8 or b.min() <= 0:
        raise ValueError('invalid shifted-demand parameters')
    a = np.maximum(a,0)
    network = _Network(edges,_matrix_from_H(external,ctx),ctx['zone_ids'])
    warm_used = warm_start is not None
    if warm_used:
        paths = list(warm_start['path_arcs'])
        path_od = warm_start['path_od'].copy()
        reachable = warm_start['reachable'].copy()
        old_f = warm_start['path_flow']
        old_q = np.bincount(path_od,weights=old_f,minlength=len(a))
        guess = a+b*np.exp(-gamma*(warm_start['costs']-reference))
        f = old_f*guess[path_od]/np.maximum(old_q[path_od],1e-200)
    else:
        shortest, raw = _shortest_paths(network,network.t0,ctx['od_pairs'])
        reachable = np.isfinite(raw)
        paths = [p for p in shortest if p is not None]
        path_od = np.flatnonzero(reachable)
        f = (a+b)[reachable].copy()
    if not paths:
        costs = np.full_like(external, ctx["u_pen"])
        q = a+b*np.exp(-gamma*(costs-reference))
        return q, costs, dict(fixed_point_residual=0., ue_gap=0., iterations=0,
            rounds=0, warm_start=warm_used), dict(path_arcs=[], path_od=path_od,
            path_flow=f, costs=costs, flow=np.zeros(network.m),
            link_cost=network.t0.copy(), reachable=reachable), []
    keys = set(zip(path_od.tolist(),paths))
    floor = b*.01
    trace, refine, total_iterations = [], False, 0
    path_threshold = min(.05,np.log1p(tolerance)/gamma)
    for r in range(1,max_rounds+1):
        incidence = _incidence_matrix(paths,network.m)
        objective = potential_factory(network,incidence,path_od,a,b,reference,gamma,reachable,floor)
        opt = minimize(objective,f,method='L-BFGS-B',jac=True,bounds=[(0,None)]*len(f),
                       options=dict(ftol=1e-12 if refine else 1e-9,
                                    gtol=1e-6 if refine else 1e-4,
                                    maxiter=1000 if refine else 500,maxls=50,maxcor=20))
        f = np.maximum(opt.x,0)
        total_iterations += opt.nit
        flow = np.asarray(incidence@f).ravel()
        q = np.bincount(path_od,weights=f,minlength=len(a))
        link_cost = network.cost(flow)
        shortest, raw = _shortest_paths(network,link_cost,ctx['od_pairs'])
        costs = np.where(np.isfinite(raw),raw,ctx['u_pen'])
        target = a+b*np.exp(-gamma*(costs-reference))
        q[~reachable] = target[~reachable]
        residual = float(np.max(np.abs(target-q)/np.maximum(external,1)))
        total_cost = float(flow@link_cost)
        gap = max(0., (total_cost-float(q[reachable]@costs[reachable]))/max(total_cost,1))
        z = q-a
        valid_domain = bool(np.all(z[reachable] > floor[reachable]))
        marginal = np.log(np.maximum(z,floor)/b)/gamma-reference
        marginal += np.minimum(z-floor,0)/(gamma*floor)
        additions=[]
        for i,p in enumerate(shortest):
            if p is not None and (i,p) not in keys and costs[i]+marginal[i] < -path_threshold:
                additions.append((i,p)); keys.add((i,p))
        trace.append(dict(round=r,residual=residual,ue_gap=gap,valid_domain=valid_domain,
                          iterations=int(opt.nit),new_paths=len(additions),paths=len(paths)))
        if residual <= tolerance and gap <= ue_tolerance and valid_domain and not additions:
            return q,costs,dict(fixed_point_residual=residual,ue_gap=gap,iterations=total_iterations,
                rounds=r,warm_start=warm_used,min_elastic_over_extension=float(np.min(z[reachable]/floor[reachable]))),dict(
                path_arcs=paths,path_od=path_od.copy(),path_flow=f,costs=costs,flow=flow,link_cost=link_cost,reachable=reachable),trace
        if not additions:
            refine=True
            if not valid_domain:
                floor *= .1
            else:
                path_threshold=max(path_threshold*.2,1e-6)
        else:
            refine=False
            paths.extend(p for _,p in additions)
            path_od=np.r_[path_od,[i for i,_ in additions]]
            f=np.r_[f,np.zeros(len(additions))]
    raise RuntimeError(f'shifted convex solver failed: {trace[-3:]}')
