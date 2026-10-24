"""Run only the finalized daily problem in doc/notes/repository_logic_map.md.

Historical problem entry points are retired; their source is preserved in .legacy/.
Natural-only demand remains an explicit training ablation, never a test environment.
"""
import argparse

from src import config as P
from src.problem.daily import BRANCHES, CURRENT_PROBLEM


METHODS = ("rl_s2v_saa64_adaptive", "rl_s2v_saa64_adaptive_roadclass", "ga", "ga_saa10_flow_insert", "compare", "rl_ablations")


def daily_cli(argv=None):
    parser = argparse.ArgumentParser(description="Daily road restoration with uncertain, gradually discovered damage")
    parser.add_argument("--setting", choices=("daily",), default="daily",
                        help="only the finalized daily problem is supported")
    parser.add_argument("--n", choices=BRANCHES, required=True,
                        help="explicit damage branch: 6, 11, 17, 23, or equal-probability mixed")
    parser.add_argument("--solve", default="rl_s2v_saa64_adaptive",
                        help="comma-separated methods: " + ", ".join(METHODS))
    parser.add_argument("--training-behavior", choices=("natural", "response7d", "both"), default="response7d")
    parser.add_argument("--seed", type=int, default=P.SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--describe", action="store_true",
                        help="show the public problem and scenario counts without training or saving results")
    parser.add_argument("--road-class-input", action=argparse.BooleanOptionalAction, default=True,
                        help="include public highway/major/local inputs in either training behavior")
    args = parser.parse_args(argv)
    methods = [name.strip() for name in args.solve.split(",")]
    # Validate the entire request before starting any method or writing outputs.
    if any(name not in METHODS for name in methods):
        parser.error("unsupported method; only finalized-problem methods are available: " + ", ".join(METHODS))
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if "rl_s2v_saa64_adaptive_roadclass" in methods and not args.road_class_input:
        parser.error("the roadclass solver requires --road-class-input")
    if "rl_ablations" in methods and (len(methods) != 1 or not args.road_class_input):
        parser.error("rl_ablations is a complete three-model experiment with road class; run it alone")

    n = "mixed" if args.n == "mixed" else int(args.n)
    if args.describe:
        import json
        from src.experiment.daily import prepare
        from src.problem.daily import public_spec, improvement_splits
        problem, splits, simulator, _, folder, _ = prepare(n)
        try:
            expanded = improvement_splits(problem, splits)
            print(json.dumps(dict(problem=public_spec(problem),
                  scenarios=dict(base_train=len(splits['train']), improvement_train=len(expanded['train']),
                                 validation=len(splits['validation']), test=len(splits['test'])),
                  output_directory=str(folder), training_started=False), indent=2))
        finally:
            simulator.close()
        return
    from src.experiment.daily import run, compare
    for method in methods:
        if method in METHODS[:2]:
            run(n, args.training_behavior, args.seed, args.workers,
                road_class_input=args.road_class_input, problem_variant=CURRENT_PROBLEM)
        elif method == "ga":
            from src.experiment.daily_ga import run as run_daily_ga
            print(run_daily_ga(n, args.seed, args.workers))
        elif method == "ga_saa10_flow_insert":
            from src.experiment.daily_ga_saa10 import run as run_daily_ga_saa10
            print(run_daily_ga_saa10(n, args.seed, args.workers,
                                    problem_variant=CURRENT_PROBLEM))
        elif method == "rl_ablations":
            from src.experiment.daily import run_rl_ablations
            print(run_rl_ablations(n, args.seed, args.workers))
        else:
            print(compare(n, problem_variant=CURRENT_PROBLEM))


if __name__ == "__main__":
    daily_cli()
