"""Small deterministic trainer checks; no real scenarios or test scores used."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.methods.daily_s2v import train, POLICY_DELIVERY


class TinyEncoder:
    node_dimension = 2
    global_dimension = 1
    A = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    deg = np.ones(2, dtype=np.float32)

    def specification(self):
        return {"test_encoder": True}


STATE = dict(x=np.array([[1., .2], [.4, .8]], dtype=np.float32),
             g=np.array([.3], dtype=np.float32), cand=(0, 1), roads=(1, 2), day=1)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def make_batch(values):
    captured = []

    def batch(split, indices, weights, epsilon, seeds):
        assert split in ("train", "validation"), "no test data may be queried"
        objective = 1.0
        if split == "validation":
            objective = values[len(captured)]
            captured.append(copy.deepcopy(weights))
        return [dict(states=[copy.deepcopy(STATE)], picks=[0], rewards=[-1.0],
                     result=dict(objective=objective, order=[1])) for _ in indices]
    return batch, captured


class PolicySelectionChecks(unittest.TestCase):
    def options(self, root):
        return dict(train_count=1, validation_count=1, seed=42, run_dir=root,
                    identity={"synthetic_check": True},
                    hp=dict(pool_n=1, batch_worlds=1, updates_per_ep=1, batch=1,
                            p=8, t_emb=1, readout_hidden=8),
                    stop_params=dict(n_val=1, ep_min=99, probe_every=1), ep_cap=3)

    def test_best_not_final_and_last_resume_state_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, weights = make_batch([3., 1., 2.])
            policy, result = train(TinyEncoder(), batch, **self.options(root))
            selected = torch.load(root / "results/model_best_validation.pt", weights_only=True)
            final = torch.load(root / "results/model_final.pt", weights_only=True)
            latest = torch.load(root / "results/training_latest.pt", weights_only=False)
            equal_tree(policy.net.state_dict(), weights[1])
            equal_tree(selected, weights[1])
            equal_tree(final, weights[2])
            equal_tree(latest["online"], final)
            self.assertTrue(any(not torch.equal(final[k], selected[k]) for k in final))
            selection = result["policy_selection"]
            self.assertEqual(selection["selected_episode"], 1)
            self.assertEqual(selection["validation_objective"], 1.)
            self.assertFalse(selection["test_used_for_selection"])
            self.assertEqual(json.loads((root / "config/training.json").read_text())["delivery"], POLICY_DELIVERY)
            self.assertEqual(json.loads((root / "results/policy_selection.json").read_text()), selection)

    def test_ties_use_earliest(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, weights = make_batch([1., 1., 2.])
            policy, result = train(TinyEncoder(), batch, **self.options(Path(tmp)))
            equal_tree(policy.net.state_dict(), weights[0])
            self.assertEqual(result["policy_selection"]["selected_episode"], 0)

    def test_pause_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "direct", Path(tmp) / "resumed"
            batch_a, _ = make_batch([3., 1., 2.])
            direct, result_a = train(TinyEncoder(), batch_a, **self.options(a))
            batch_b, _ = make_batch([3., 1., 2.])
            _, pause = train(TinyEncoder(), batch_b, **self.options(b), pause_after=1)
            self.assertEqual(pause["outcome"], "screening_pause")
            self.assertFalse((b / "results/policy_selection.json").exists())
            resumed, result_b = train(TinyEncoder(), batch_b, **self.options(b))
            equal_tree(direct.net.state_dict(), resumed.net.state_dict())
            equal_tree(result_a["policy_selection"], result_b["policy_selection"])
            ca = torch.load(a / "results/training_latest.pt", weights_only=False)
            cb = torch.load(b / "results/training_latest.pt", weights_only=False)
            for key in ("online", "target", "optimizer", "rng", "torch_rng", "replay",
                        "priorities", "exemplars", "stop_state", "best_weights"):
                equal_tree(ca[key], cb[key])

    def test_no_validation_fails_instead_of_using_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, _ = make_batch([float("nan")] * 3)
            with self.assertRaisesRegex(RuntimeError, "finite validation"):
                train(TinyEncoder(), batch, **self.options(Path(tmp)))

    def test_new_rule_does_not_change_learning(self):
        # In the deferred sandbox compare the unchanged live trainer with the
        # proposed version, using identical synthetic transitions and RNG seeds.
        original = Path(__file__).resolve().parents[4] / "src/methods/daily_s2v.py"
        if not original.is_file():
            self.skipTest("live-vs-staged comparison is only available before deployment")
        module_spec = importlib.util.spec_from_file_location("original_daily_trainer", original)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "old", Path(tmp) / "new"
            ba, _ = make_batch([3., 1., 2.])
            bb, _ = make_batch([3., 1., 2.])
            module.train(TinyEncoder(), ba, **self.options(a))
            train(TinyEncoder(), bb, **self.options(b))
            ca = torch.load(a / "results/training_latest.pt", weights_only=False)
            cb = torch.load(b / "results/training_latest.pt", weights_only=False)
            for key in ("online", "target", "optimizer", "rng", "torch_rng", "replay",
                        "priorities", "exemplars", "stop_state", "best_weights"):
                equal_tree(ca[key], cb[key])
            for ra, rb in zip(ca["records"], cb["records"]):
                for key in ra:
                    if key not in ("seconds", "cumulative_seconds"):
                        equal_tree(ra[key], rb[key])


if __name__ == "__main__":
    unittest.main()
