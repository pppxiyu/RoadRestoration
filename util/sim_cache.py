"""
Persistent, cross-run, cross-method cache of per-slot accessibility terms, under
cache/sim_cache/ at the project root.

WHY IT IS SOUND. The slot term g_k that dominates evaluation cost (one user-equilibrium solve
per slot) is a deterministic function of two things: the damage-TRAJECTORY prefix up to slot k,
and the fixed problem definition. The trajectory prefix is what the key encodes -- each segment's
completion slot capped at k+1, the exact key util.rl._evaluate_prefix_cached already memoizes on
within a run (capping keeps "completed at k" distinct from "still damaged"; the demand shortfall
D_k carries memory, so the whole prefix matters and no per-slot state short of it would be a
valid key). Everything else lives in the FINGERPRINT.

WHY IT CANNOT SERVE A STALE TERM. Entries are stored under a fingerprint that hashes the problem
definition at both levels: the parameters (RHO, KAPPA, UPEN_FACTOR, both UE tolerance regimes)
AND the derived arrays the slot solve actually reads -- the network files, the disrupted-segment
list in key order, B, H0, severity, baseline travel times, the disconnection penalty. Hashing
the arrays as well as the parameters means that even a change that bypasses config (edited input
data, a different instance selection, a code change to how B is built) lands in a DIFFERENT
cache directory rather than colliding with old entries. Each fingerprint directory carries a
human-readable fingerprint.json naming what produced it, so the cache is auditable. A change in
anything the fingerprint covers simply starts an empty cache: the failure mode is a cold start,
never a wrong number.

WHAT IS DELIBERATELY OUTSIDE THE FINGERPRINT. MU and F1_ACTIVE_ONLY aggregate the per-slot terms
AFTER the cache; C_MAX and the duration model shape which trajectories occur, not what a given
trajectory's slot term is. Including them would only shrink hit rates without adding safety.

Robustness: every cache operation is wrapped so that any storage failure (locked file, corrupt
db, missing folder) degrades to a cache miss -- the evaluation then simply solves UE live. The
cache can accelerate an evaluation; it can never break one. Writes use INSERT OR IGNORE under
SQLite WAL, so concurrent workers (the GA's evaluation pool) cannot corrupt or duplicate.

Maintenance entry points (the cache subsystem's own, per the main.py/config.py control rule):
  python -m util.sim_cache --seed     seed the current-fingerprint cache from results already on
                                      disk (only from runs whose run_meta matches the current
                                      objective/UE parameters and instance -- anything else is
                                      skipped, never guessed)
  python -m util.sim_cache --verify   re-evaluate every canonical delivered schedule against the
                                      cache and assert the F on disk is reproduced exactly
  python -m util.sim_cache --stats    entry counts per fingerprint
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config as P

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / "cache" / "sim_cache"


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def _sha(b):
    return hashlib.sha1(b).hexdigest()


def _arr(a):
    return _sha(np.ascontiguousarray(a).tobytes())


def fingerprint(ctx, toy_dir):
    """The identity of everything, other than the trajectory key, that determines a slot term.
    Returns (fp_hex, manifest_dict); the manifest is what fingerprint.json records."""
    toy_dir = Path(toy_dir)
    data_files = {f.name: _sha(f.read_bytes()) for f in sorted(toy_dir.glob("*.tntp"))}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_files": data_files,
        # the disrupted list IN ORDER: the key's completion vector is positional, so the order
        # is part of the key's meaning, not just the set
        "disrupted": [[int(e), int(u), int(v), float(s)] for (e, u, v, s) in ctx["disrupted"]],
        "arrays": {"B": _arr(ctx["B"]), "H0": _arr(ctx["H0"]),
                   "severity": _arr(ctx["severity_vec"]), "baseline_u": _arr(ctx["baseline_u"]),
                   "u_pen": float(ctx["u_pen"])},
        "params": {"RHO": P.RHO, "KAPPA": P.KAPPA, "UPEN_FACTOR": P.UPEN_FACTOR,
                   "UE_RGAP": P.UE_RGAP, "UE_MAX_ITER": P.UE_MAX_ITER,
                   "UE_WARM_START": P.UE_WARM_START,
                   "UE_RGAP_DEF": P.UE_RGAP_DEF, "UE_MAX_ITER_DEF": P.UE_MAX_ITER_DEF},
    }
    ident = {k: manifest[k] for k in ("data_files", "disrupted", "arrays", "params")}
    fp = _sha(json.dumps(ident, sort_keys=True).encode())[:16]
    return fp, manifest


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class SimCache:
    """Slot-term store for one fingerprint. get/put never raise: any storage problem degrades to
    a miss (get -> None) or a dropped write, and the evaluation proceeds live."""

    def __init__(self, fp, manifest=None):
        self.fp = fp
        self.dir = CACHE_ROOT / fp
        self._con = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            mpath = self.dir / "fingerprint.json"
            if manifest is not None and not mpath.exists():
                mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._con = sqlite3.connect(self.dir / "slots.sqlite", timeout=10)
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute("CREATE TABLE IF NOT EXISTS slots ("
                              "k INTEGER NOT NULL, comp TEXT NOT NULL, g REAL NOT NULL, "
                              "PRIMARY KEY (k, comp))")
            self._con.commit()
        except Exception as exc:
            print(f"[sim_cache] disabled for this run ({type(exc).__name__}: {exc})", flush=True)
            self._con = None

    @staticmethod
    def _ser(key):
        k, comp = key
        return int(k), "-".join(map(str, comp))

    def get(self, key):
        if self._con is None:
            return None
        try:
            k, comp = self._ser(key)
            row = self._con.execute("SELECT g FROM slots WHERE k=? AND comp=?",
                                    (k, comp)).fetchone()
            return None if row is None else float(row[0])
        except Exception:
            return None

    def put(self, key, g):
        if self._con is None:
            return
        try:
            k, comp = self._ser(key)
            self._con.execute("INSERT OR IGNORE INTO slots (k, comp, g) VALUES (?, ?, ?)",
                              (k, comp, float(g)))
            self._con.commit()
        except Exception:
            pass

    def put_many(self, items):
        if self._con is None:
            return
        try:
            rows = [(*self._ser(k), float(g)) for k, g in items]
            self._con.executemany("INSERT OR IGNORE INTO slots (k, comp, g) VALUES (?, ?, ?)", rows)
            self._con.commit()
        except Exception:
            pass

    def n_entries(self):
        try:
            return int(self._con.execute("SELECT COUNT(*) FROM slots").fetchone()[0])
        except Exception:
            return -1


def for_ctx(ctx, toy_dir=None):
    """The SimCache bound to this evaluation context, or None when P.SIM_CACHE is off. Cached on
    the ctx itself so the fingerprint is computed once per context per process. `toy_dir` is only
    needed the first time; the evaluators pass the directory recorded on the ctx."""
    if not getattr(P, "SIM_CACHE", False):
        return None
    sc = ctx.get("_sim_cache")
    if sc is None:
        td = toy_dir or ctx.get("toy_dir")
        if td is None:
            return None
        fp, manifest = fingerprint(ctx, td)
        sc = SimCache(fp, manifest)
        ctx["_sim_cache"] = sc
    return sc


# --------------------------------------------------------------------------- #
# Maintenance: seed from results already on disk, verify against them
# --------------------------------------------------------------------------- #
def _iter_canonical_runs(N):
    """Yield (method, scale_dir) for every method folder holding an optima file at scale N."""
    from util.provenance import solver_dir
    groups = {f"02-baselines/{solver_dir('rule-based')}": ("flow", "demand", "ratio"),
              f"02-baselines/{solver_dir('ga')}": ("ga",),
              f"02-baselines/{solver_dir('pretrain_milp')}": ("milp",),
              f"03-rl/{solver_dir('rl_nominal')}": ("rl_nominal",)}
    for rel, methods in groups.items():
        d = ROOT / "outputs" / rel / f"n{N}"
        for m in methods:
            if (d / "results" / f"{m}_optima.csv").exists():
                yield m, d


def _run_meta_matches(scale_dir, method, manifest):
    """A run may be seeded only if its recorded objective/UE parameters equal the current
    fingerprint's. Runs without a readable meta, or from other parameter regimes, are skipped --
    the cache must be under-seeded rather than ever wrongly seeded."""
    for name in (f"{method}_run_meta.json", "run_meta.json"):
        p = scale_dir / "config" / name
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8")).get("objective", {})
            except Exception:
                return False, "unreadable run_meta"
            want = manifest["params"]
            for kk, mine in (("RHO", want["RHO"]), ("KAPPA", want["KAPPA"]),
                             ("UPEN_FACTOR", want["UPEN_FACTOR"]), ("UE_RGAP", want["UE_RGAP"]),
                             ("UE_MAX_ITER", want["UE_MAX_ITER"]),
                             ("UE_WARM_START", want["UE_WARM_START"])):
                if kk in obj and obj[kk] != mine:
                    return False, f"{kk} mismatch ({obj[kk]} vs {mine})"
            return True, "ok"
    return False, "no run_meta"


def seed(N=None, toy_dir=None):
    """Populate the current-fingerprint cache from optima+slots files already on disk.

    The values come from {m}_slots.csv (the recorded per-slot terms, which reconstitute the
    reported F exactly); the keys are rebuilt from each scenario's recorded start slots and
    durations through the SAME capped-completion formula the evaluators use. A run is used only
    when its run_meta matches the current parameters. Always run --verify afterwards: it replays
    every canonical schedule against the cache and proves the F on disk is reproduced."""
    import pandas as pd
    from util.evaluate import build_context
    from util.oracle import select_oracle_instance
    N = P.N_DISRUPTED_ORACLE if N is None else N
    toy_dir = toy_dir or (ROOT / "data" / "siouxfalls_toy")
    dis = select_oracle_instance(toy_dir, N)
    ctx = build_context(toy_dir, dis, ue_cores=1)
    fp, manifest = fingerprint(ctx, toy_dir)
    sc = SimCache(fp, manifest)
    segments = sorted(int(r) for r in dis["edge_id"])
    dis_order = [int(e) for (e, _, _, _) in ctx["disrupted"]]
    total = 0
    for m, d in _iter_canonical_runs(N):
        ok, why = _run_meta_matches(d, m, manifest)
        if not ok:
            print(f"  [seed] SKIP {m} @ {d.name}: {why}", flush=True)
            continue
        opt = pd.read_csv(d / "results" / f"{m}_optima.csv")
        spath = d / "log" / f"{m}_slots.csv"
        if not spath.exists():
            print(f"  [seed] SKIP {m}: no slots.csv", flush=True)
            continue
        slots = pd.read_csv(spath)
        items = []
        for _, row in opt.iterrows():
            durs = dict(zip(segments, (int(x) for x in str(row["durations"]).split("-"))))
            start = {e: int(row[f"start_{e}"]) for e in segments}
            comp = tuple(start[e] + durs[e] for e in dis_order)
            gmap = {int(r.slot): float(r.g)
                    for r in slots[slots.scenario == row["scenario"]].itertuples()}
            for k, g in gmap.items():
                items.append(((k, tuple(min(c, k + 1) for c in comp)), g))
        sc.put_many(items)
        total += len(items)
        print(f"  [seed] {m}: {len(items)} slot terms", flush=True)
    print(f"seeded {total} terms -> {sc.dir}  (entries now {sc.n_entries()})", flush=True)
    return fp


def verify(N=None, toy_dir=None):
    """Replay every canonical delivered schedule through the cached evaluator and assert the F on
    disk is reproduced. A wrong seeded VALUE would change F (caught); a wrong seeded KEY would
    surface as cache misses (reported via n_ue > 0). Exact-zero UE solves plus matching F is the
    proof the cache serves precisely the recorded evaluations."""
    import pandas as pd
    from util.evaluate import build_context
    from util.oracle import compute_horizon, select_oracle_instance
    from util.rl import _evaluate_prefix_cached
    from util.scenarios import sample_scenarios
    N = P.N_DISRUPTED_ORACLE if N is None else N
    toy_dir = toy_dir or (ROOT / "data" / "siouxfalls_toy")
    dis = select_oracle_instance(toy_dir, N)
    ctx = build_context(toy_dir, dis, ue_cores=1)
    ctx["toy_dir"] = str(toy_dir)
    segments = sorted(int(r) for r in dis["edge_id"])
    scen = sample_scenarios(dis, P.M_SCENARIOS, P.SEED)
    T = compute_horizon(segments, scen)
    worst = 0.0
    total_ue = 0
    for m, d in _iter_canonical_runs(N):
        opt = pd.read_csv(d / "results" / f"{m}_optima.csv")
        col = "F_milp" if "F_milp" in opt.columns else "F"
        ue_m, diff_m = 0, 0.0
        for _, row in opt.iterrows():
            sc_idx = int(row["scenario"])
            durs = scen[sc_idx]
            start = {e: int(row[f"start_{e}"]) for e in segments}
            res = _evaluate_prefix_cached(start, durs, T, ctx, {})
            ue_m += res["n_ue"]
            diff_m = max(diff_m, abs(res["F"] - float(row[col])))
        total_ue += ue_m
        worst = max(worst, diff_m)
        print(f"  [verify] {m}: max|F - disk| = {diff_m:.2e}   UE solves = {ue_m}", flush=True)
    print(f"VERIFY {'PASS' if worst < 1e-9 else 'FAIL'}: worst F deviation {worst:.2e}, "
          f"total live UE solves {total_ue} (0 = every slot served from cache)", flush=True)
    return worst < 1e-9


def stats():
    for d in sorted(CACHE_ROOT.glob("*")):
        if (d / "slots.sqlite").exists():
            sc = SimCache(d.name)
            mp = d / "fingerprint.json"
            created = ""
            if mp.exists():
                try:
                    created = json.loads(mp.read_text(encoding="utf-8")).get("created_at", "")
                except Exception:
                    pass
            print(f"  {d.name}: {sc.n_entries()} slot terms  ({created})", flush=True)


if __name__ == "__main__":
    import sys
    if "--seed" in sys.argv:
        seed()
    elif "--verify" in sys.argv:
        raise SystemExit(0 if verify() else 1)
    else:
        stats()
