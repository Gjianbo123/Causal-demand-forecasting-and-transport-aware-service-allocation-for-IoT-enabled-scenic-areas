"""Leakage, topology, serialization, and output checks for the new forecasters."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from iotexp.scenic_forecast import (TemporalForecaster, build_windows, edge_arrays,
    fit_training_stats, forecast_config, forecast_window, load_forecaster, prepare_forecaster,
    split_day_indices)
from iotexp.telemetry import Window


def tiny_data():
    rng = np.random.default_rng(44)
    times = np.arange(-3, 9)
    values = rng.poisson(8+3*np.sin(times[None, :, None]/3)+np.arange(3)[None, None], (10, 12, 3)).astype(np.float32)
    observations = np.stack([values]+[np.maximum(values-(i+1), 0) for i in range(4)], axis=-1)
    return {"inflow": values, "observations": observations, "times": times, "split_days": [6, 2, 2],
            "adjacency": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            "edge_index": np.array([[0, 1], [1, 2]]), "edge_lengths_m": np.array([50, 100])}


def tiny_config():
    return {"scenic": {"history": 3, "horizons": [1, 2], "episode_steps": 8},
            "forecaster": {"seeds": [0], "epochs": 2, "patience": 2, "batch_size": 16,
                           "hidden": 8, "selected_model": "edge_stgru", "selected_seed": 0}}


class ForecastTests(unittest.TestCase):
    def test_stats_exclude_validation_and_test(self):
        data = tiny_data()
        days = split_day_indices(data)["train"]
        before = fit_training_stats(data, days, 8)
        data["observations"][6:] = 100000
        after = fit_training_stats(data, days, 8)
        np.testing.assert_array_equal(before.mean, after.mean)
        np.testing.assert_array_equal(before.scale, after.scale)

    def test_window_alignment_and_split(self):
        data, cfg = tiny_data(), forecast_config(tiny_config())
        days = split_day_indices(data)
        stats = fit_training_stats(data, days["train"], 8)
        w = build_windows(data, days["test"], cfg, stats)
        self.assertEqual(w["x"].shape, (14, 3, 3))
        self.assertEqual(set(w["day"]), {8, 9})
        np.testing.assert_array_equal(w["y"][0], data["inflow"][8, [4, 5]].T)
        np.testing.assert_array_equal(w["history_times"][0], [-2, -1, 0])

    def test_only_physical_neighbors_influence_node(self):
        data = tiny_data()
        mask, lengths = edge_arrays(data)
        torch.manual_seed(3)
        network = TemporalForecaster("edge_stgru", 8, 2, mask, lengths)
        x = torch.randn(1, 3, 3, requires_grad=True)
        output = network(x, torch.zeros(1, 3, 2))
        output[0, 0].sum().backward()
        self.assertGreater(float(x.grad[0, :, 1].abs().sum()), 0)
        self.assertEqual(float(x.grad[0, :, 2].abs().sum()), 0)

    def test_all_models_serialization_and_causal_api(self):
        torch.set_num_threads(1)
        data, cfg = tiny_data(), tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            stats, model = prepare_forecaster(data, cfg, directory)
            loaded_stats, loaded_model = load_forecaster(directory)
            window = Window(data["observations"][8, 1:4], np.ones((3, 3)), np.zeros((3, 3)),
                            np.zeros((3, 3)), np.zeros((3, 3)), np.arange(-2, 1), 0)
            actual = forecast_window(model, window, stats)
            restored = forecast_window(loaded_model, window, loaded_stats)
            np.testing.assert_allclose(actual, restored, atol=1e-6)
            self.assertEqual(actual.shape, (3, 2))
            self.assertTrue(np.isfinite(actual).all() and (actual >= 0).all())
            self.assertTrue((Path(directory)/"forecast"/"daily_metrics.csv").is_file())
            self.assertTrue((Path(directory)/"forecast"/"lstm_seed0_test_predictions.npz").is_file())


if __name__ == "__main__":
    unittest.main()
