"""Equal-probability mixture of the four existing damage-count settings.

The random damage count is environment truth, not an extra policy input.
Each scale retains its fixed public subset and original conditional sampler.
"""
from dataclasses import replace
import numpy as np

from src.problem.daily import build_problem, sample_split, public_spec

SCALES = (6, 11, 17, 23)
PROBABILITIES = (0.25, 0.25, 0.25, 0.25)


def build_mixed_problem(seed=42):
    components = {n: build_problem(n, seed, variant="poi_unmet") for n in SCALES}
    base = dict(components[11])
    # All simulation rules/network/behavior remain those of poi_unmet. Only
    # the scenario distribution differs; its full contract enters identity.
    base.update(n="mixed", variant="poi_mixed_n", components=components,
                public_roads=tuple(sorted({e for p in components.values() for e in p['public_roads']})))
    return base


def sample_mixed_split(problem, count, seed, name):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2401]))
    scales = rng.choice(SCALES, size=int(count), p=PROBABILITIES)
    output = [None] * int(count)
    for n in SCALES:
        locations = np.flatnonzero(scales == n)
        if not len(locations):
            continue
        conditional_seed = int(np.random.SeedSequence([seed, n, 2402]).generate_state(1)[0])
        worlds = sample_split(problem['components'][n], len(locations), conditional_seed, name)
        for index, world in zip(locations, worlds):
            output[int(index)] = replace(world, scenario_id=f"{name}_{index:03d}")
    return output


def mixed_public_spec(problem):
    spec = public_spec(problem['components'][11])
    spec.update(n_disrupted=list(SCALES), variant="poi_mixed_n",
        public_roads=None, hidden_quotas=None,
        count_distribution=dict(values=SCALES, probabilities=PROBABILITIES,
            sampling="independent categorical draw for every scenario in train, validation and test",
            conditional_labels="independent per-scale LHS; original damage/discovery rules",
            policy_information="true total n is not supplied to either policy"),
        scale_settings={str(n): dict(public_roads=p['public_roads'], hidden_quotas=p['hidden_quotas'])
                        for n, p in problem['components'].items()})
    return spec
