import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iotexp.forecast_supplement import (
    NEURAL_MODELS, CANDIDATES, make_model, setup_runtime,
    verify_selection, ITransformer, _scores,
)


class ForecastSupplementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"history": 12, "horizons": [1, 2, 4, 6], "period": 72, "device": "cpu", "torch_threads": 2}
        setup_runtime(self.cfg)
        adjacency = np.eye(4, k=1)+np.eye(4, k=-1)
        self.data = {"adjacency": adjacency, "edge_index": np.array([[0, 1], [1, 2], [2, 3]]),
                     "edge_lengths_m": np.array([1., 2., 4.])}
        self.x = torch.randn(3, 12, 4)
        self.clock = torch.randn(3, 12, 2)
        self.origin = torch.tensor([0, 8, 66])

    def test_all_architectures_have_valid_gradient_and_shape(self):
        for name in NEURAL_MODELS:
            with self.subTest(name=name):
                model = make_model(name, CANDIDATES[0], self.data, self.cfg)
                output = model(self.x, self.clock, self.origin)
                self.assertEqual(tuple(output.shape), (3, 4, 4))
                loss = (output-torch.randn_like(output)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_graph_ablations_preserve_local_information_boundary(self):
        for name in ["edge_stgru", "uniform_neighbor", "no_edge_length"]:
            model = make_model(name, CANDIDATES[0], self.data, self.cfg).eval()
            before = model(self.x, self.clock, self.origin)
            changed = self.x.clone()
            changed[:, :, 3] += 100
            after = model(changed, self.clock, self.origin)
            torch.testing.assert_close(before[:, 0], after[:, 0], rtol=0, atol=0)

    def test_edge_length_ablation_does_not_use_lengths(self):
        other = {**self.data, "edge_lengths_m": np.array([100., 1., 2.])}
        for name in ["no_edge_length", "uniform_neighbor"]:
            first = make_model(name, CANDIDATES[0], self.data, self.cfg).eval()
            second = make_model(name, CANDIDATES[0], other, self.cfg).eval()
            second.load_state_dict(first.state_dict())
            torch.testing.assert_close(first(self.x, self.clock, self.origin), second(self.x, self.clock, self.origin))

    def test_matched_ablation_changes_only_the_declared_aggregation(self):
        for name in ["uniform_neighbor", "no_edge_length"]:
            torch.manual_seed(7)
            full = make_model("edge_stgru", CANDIDATES[1], self.data, self.cfg).eval()
            torch.manual_seed(7)
            ablation = make_model(name, CANDIDATES[1], self.data, self.cfg).eval()
            for key, value in full.core.recurrent.state_dict().items():
                torch.testing.assert_close(value, ablation.recurrent.state_dict()[key], rtol=0, atol=0)
            for key, value in full.core.head.state_dict().items():
                torch.testing.assert_close(value, ablation.head.state_dict()[key], rtol=0, atol=0)
            with torch.no_grad():
                if name == "uniform_neighbor":
                    full.core.edge_scores.zero_()
                else:
                    full.core.edge_scores[2] = 0.
            torch.testing.assert_close(full(self.x, self.clock, self.origin),
                                       ablation(self.x, self.clock, self.origin), rtol=1e-6, atol=1e-6)

    def test_itransformer_is_equivariant_to_variate_permutation(self):
        model = ITransformer(32, 4).eval()
        permutation = [2, 0, 3, 1]
        before = model(self.x, self.clock, self.origin)
        after = model(self.x[:, :, permutation], self.clock, self.origin)
        torch.testing.assert_close(before[:, permutation], after, rtol=1e-5, atol=1e-5)

    def test_checkpoint_round_trip_for_each_architecture(self):
        for name in NEURAL_MODELS:
            model = make_model(name, CANDIDATES[0], self.data, self.cfg).eval()
            clone = make_model(name, CANDIDATES[0], self.data, self.cfg).eval()
            clone.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(self.x, self.clock, self.origin), clone(self.x, self.clock, self.origin), rtol=0, atol=0)

    def test_graph_wavenet_storage_layout_preserves_prediction(self):
        optimized = make_model("graph_wavenet", CANDIDATES[0], self.data, self.cfg).eval()
        contiguous = make_model("graph_wavenet", CANDIDATES[0], self.data, self.cfg).to(memory_format=torch.contiguous_format).eval()
        contiguous.load_state_dict(optimized.state_dict())
        torch.testing.assert_close(optimized(self.x, self.clock, self.origin), contiguous(self.x, self.clock, self.origin), rtol=1e-5, atol=1e-6)

    def test_valid_zero_targets_are_not_masked(self):
        result = _scores(np.array([0., 2.]), np.array([10., 2.]))
        self.assertEqual(result["mae"], 5.)
        self.assertEqual(result["observations"], 2)
        self.assertEqual(result["wape_pct"], 500.)

    def test_final_evaluation_requires_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "forbidden"):
                verify_selection(Path(directory))


if __name__ == "__main__":
    unittest.main()
