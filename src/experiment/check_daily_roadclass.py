"""Road-class input ablation checks; no official training or test evaluation."""
import json
from pathlib import Path
import types
import unittest
import zipfile

import numpy as np
import torch

from src.experiment.check_daily import PolicyChecks
from src.methods.daily_s2v import DailyEncoder, DailyQ


class RoadClassChecks(PolicyChecks):
    def test_extended_training_and_resume(self):
        original = self.encoder
        try:
            self.encoder = DailyEncoder(self.problem, self.context, road_class_input=True)
            PolicyChecks.test_network_and_resumable_training(self)
        finally:
            self.encoder = original

    def test_original_features_and_public_categories(self):
        encoder = DailyEncoder(self.problem, self.context, road_class_input=True)
        observation = self.observation(70)
        original = self.encoder.encode(observation)
        extended = encoder.encode(observation)
        np.testing.assert_array_equal(original["x"], extended["x"][:, :14])
        np.testing.assert_array_equal(original["g"], extended["g"])
        self.assertEqual(original["cand"], extended["cand"])
        self.assertEqual(extended["x"].shape, (38, 17))
        np.testing.assert_array_equal(extended["x"][:, 14:].sum(axis=1), np.ones(38))
        classes = self.problem["edges"].set_index("edge_id").road_class.to_dict()
        for i, edge in enumerate(encoder.ids):
            self.assertEqual(np.argmax(extended["x"][i, 14:]),
                             ("highway", "major", "local").index(classes[edge]))
        hidden = [encoder.idx[e] for e in encoder.ids if e not in observation.candidates]
        np.testing.assert_array_equal(extended["x"][hidden, 14:].sum(axis=1), np.ones(len(hidden)))

    def test_new_columns_receive_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        encoder = DailyEncoder(self.problem, self.context, road_class_input=True)
        policy = DailyQ(encoder)
        value = policy(encoder.encode(self.observation()))
        self.assertTrue(torch.isfinite(value).all())
        value.square().mean().backward()
        gradient = policy.net.th1.weight.grad[:, 14:]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient.abs().sum(dim=0) > 0))
        original = DailyQ(self.encoder)
        self.assertEqual(sum(p.numel() for p in policy.net.parameters()) -
                         sum(p.numel() for p in original.net.parameters()), 3 * policy.hp["p"])

    def test_archived_baseline_encoder_and_network_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        baseline = root / "outputs/experiments/n11_discovery_response7d_1day_lhs50_s42_e4a9d3c3/data/methods/rl_s2v_saa64_adaptive_train_response7d_seed42"
        with zipfile.ZipFile(baseline / "config/source_snapshot.zip") as archive:
            source = archive.read("src/methods/daily_s2v.py").decode("utf-8")
        module = types.ModuleType("archived_daily_s2v")
        exec(compile(source, "archived_daily_s2v.py", "exec"), module.__dict__)
        archived = module.DailyEncoder(self.problem, self.context)
        self.assertEqual(archived.specification(), self.encoder.specification())
        a, b = archived.encode(self.observation()), self.encoder.encode(self.observation())
        np.testing.assert_array_equal(a["x"], b["x"])
        np.testing.assert_array_equal(a["g"], b["g"])
        torch.manual_seed(42)
        old = module.DailyQ(archived)
        torch.manual_seed(42)
        new = DailyQ(self.encoder)
        for key, weight in old.net.state_dict().items():
            self.assertTrue(torch.equal(weight, new.net.state_dict()[key]))
        self.assertTrue(torch.equal(old(a), new(b)))
        contract = json.loads((baseline / "config/training.json").read_text())
        self.assertEqual(new.hp, contract["hyperparameters"])


if __name__ == "__main__":
    unittest.main()
