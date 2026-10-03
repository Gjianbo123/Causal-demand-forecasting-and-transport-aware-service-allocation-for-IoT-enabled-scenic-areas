import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from iotexp.analysis import signed_rank, holm, check_matrix
from iotexp.demo import DemoBackend, repair_allocation, allocation_is_feasible
from iotexp.engine import Session, generalized_advantage, make_policy, ppo_update
from iotexp.fusion import HorizonFusion, ActorCritic
from iotexp.metrics import Metrics, forecast_errors
from iotexp.telemetry import TrainingStats, TelemetryGateway, Condition


ROOT = Path(__file__).resolve().parents[1]


def simple_stats():
    return TrainingStats(np.zeros((2, 2)), np.ones((2, 2)), np.full((24, 2, 2), 7.0))


class TelemetryTests(unittest.TestCase):
    def test_nominal_and_missing_fallback(self):
        nominal = TelemetryGateway(simple_stats(), Condition(), 1, 2)
        missing = TelemetryGateway(simple_stats(), Condition(missing=1), 1, 2)
        for t in (0, 1):
            for gateway in (nominal, missing):
                gateway.submit(t, np.full((2, 2), t + 10.0))
        np.testing.assert_array_equal(nominal.window(1).values[:, 0, 0], [10, 11])
        np.testing.assert_array_equal(missing.window(1).values, 7)
        self.assertFalse(missing.window(1).mask.any())
        self.assertTrue(np.isnan(missing.window(1).source_event).all())

    def test_delay_is_causal_and_event_time_is_preserved(self):
        g = TelemetryGateway(simple_stats(), Condition(delay=2), 9, 3)
        for t in range(-3, 3):
            g.submit(t, np.full((2, 2), 10.0 + t))
        w = g.window(2)
        np.testing.assert_array_equal(w.grid, [0, 1, 2])
        np.testing.assert_array_equal(w.source_event[:, 0], [0, 0, 0])
        np.testing.assert_array_equal(w.age[:, 0], [0, 1, 2])
        self.assertTrue(np.nanmax(w.source_receipt) <= 2)
        snapshot = w.values.copy()
        g.submit(3, np.full((2, 2), 1e6))
        np.testing.assert_array_equal(g.window(2).values, snapshot)
        np.testing.assert_array_equal(w.values, snapshot)

    def test_corruption_stream_is_independent_and_paired(self):
        a = TelemetryGateway(simple_stats(), Condition(noise=.2), 123, 1)
        b = TelemetryGateway(simple_stats(), Condition(noise=.2), 123, 1)
        a.submit(0, np.full((2, 2), 100))
        np.random.default_rng(42).normal(size=12345)
        b.submit(0, np.full((2, 2), 200))
        np.testing.assert_allclose(a.window(0).values - 100, b.window(0).values - 200, atol=1e-5)
        np.testing.assert_array_equal(a.arrays()["packet_noise_draw"], b.arrays()["packet_noise_draw"])

    def test_stats_zero_variance_has_documented_fallback(self):
        s = TrainingStats.fit(np.ones((5, 2, 2)), np.arange(5), 24)
        self.assertTrue((s.scale > 0).all())
        np.testing.assert_array_equal(s.normalize(np.ones((2, 2))), 0)

    def test_duplicate_event_rejected(self):
        g = TelemetryGateway(simple_stats(), Condition(), 0, 1)
        g.submit(0, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            g.submit(0, np.ones((2, 2)))


class FusionTests(unittest.TestCase):
    def test_all_modes_shape_and_attention_normalization(self):
        torch.manual_seed(3)
        x = torch.randn(2, 5, 4)
        for mode in ("F0", "F1", "F2", "F3", "F4"):
            f = HorizonFusion(mode, 3, 4, 8)
            y, weights = f(x)
            self.assertEqual(y.shape, (2, 3, 5, 8))
            if weights is not None:
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 5))
            if mode == "F3":
                torch.testing.assert_close(weights[:, 0], weights[:, 1])
            if mode == "F4":
                (y.square().mean()).backward()
                self.assertIsNotNone(f.query.grad)
                self.assertGreater(float(f.query.grad.abs().sum()), 0)

    def test_reactive_has_no_forecast_dependency(self):
        model = ActorCritic("F0", 7, 3, 2, 16, 4, seed=5)
        state = torch.ones(2, 7)
        a = model.act(state, torch.zeros(2, 3, 4), deterministic=True)
        b = model.act(state, torch.ones(2, 3, 4) * 999, deterministic=True)
        torch.testing.assert_close(a[0], b[0])

    def test_common_backbone_initialization_is_paired(self):
        a = ActorCritic("F1", 7, 3, 2, 16, 4, seed=5)
        b = ActorCritic("F4", 7, 3, 2, 16, 4, seed=5)
        for x, y in zip(a.state_net.parameters(), b.state_net.parameters()):
            torch.testing.assert_close(x, y)
        for x, y in zip(a.fusion.embedding.parameters(), b.fusion.embedding.parameters()):
            torch.testing.assert_close(x, y)

    def test_raw_action_likelihood_is_consistent(self):
        model = ActorCritic("F4", 7, 3, 2, 16, 4, seed=9)
        state, forecast = torch.randn(2, 7), torch.randn(2, 3, 4)
        raw, old_logp, _, _ = model.act(state, forecast, torch.Generator().manual_seed(3))
        new_logp = model(state, forecast)[0].log_prob(raw)
        torch.testing.assert_close((new_logp-old_logp).exp(), torch.ones(2))

    def test_transfer_tensor_action_head(self):
        model = ActorCritic("F4", 7, 3, 2, 16, 4, action_shape=(2, 3, 3))
        raw, logp, value, _ = model.act(torch.ones(1, 7), torch.ones(1, 3, 4))
        self.assertEqual(raw.shape, (1, 2, 3, 3))
        self.assertEqual(logp.shape, (1,))


class DynamicsTests(unittest.TestCase):
    def test_repair_conserves_resources_respects_limits_and_is_deterministic(self):
        totals = np.array([7, 4])
        allowed = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        holding = np.array([[3, 3, 3], [2, 2, 2]])
        raw = np.array([[100, 0, 0], [0, 0, 100.]])
        candidate, executed = repair_allocation(raw, totals, allowed, holding)
        self.assertFalse(allocation_is_feasible(candidate, totals, allowed, holding))
        self.assertTrue(allocation_is_feasible(executed, totals, allowed, holding))
        np.testing.assert_array_equal(executed, repair_allocation(raw, totals, allowed, holding)[1])
        with self.assertRaises(ValueError):
            repair_allocation(raw, totals, allowed, holding * 0)

    def test_terminal_gae_does_not_cross_episodes(self):
        adv, returns = generalized_advantage([1, 2, 100], [.5, .6, 0], [.6, 0, 0],
                                             [False, True, True], 1, 1)
        np.testing.assert_allclose(adv, [2.5, 1.4, 100])
        np.testing.assert_allclose(returns, [3, 2, 100])

    def test_forecast_targets_stay_at_current_time_plus_horizon(self):
        result = forecast_errors({2: np.array([[4., 6.]])}, {3: np.array([4.]), 4: np.array([6.])}, [1, 2])
        self.assertEqual(result["forecast_mae_h1"], 0)
        self.assertEqual(result["forecast_mae_h2"], 0)

    def test_demo_counts_arrivals_and_overload_separately(self):
        config = json.loads((ROOT / "configs/demo_quick.json").read_text())
        env = DemoBackend(config).make_env("test", 90)
        env.reset()
        metric = Metrics()
        for _ in range(config["demo"]["episode_steps"]):
            _, _, _, audit = env.step(np.zeros((2, 6)))
            metric.add(audit)
        r = metric.result()
        self.assertEqual(r["served_count"] + r["unserved_count"], r["arrivals_count"])
        self.assertEqual(r["executable_infeasible_pct"], 0)
        self.assertGreater(r["visitor_overload_pct"], 0)


class StatisticsTests(unittest.TestCase):
    def test_exact_p_with_all_positive_differences(self):
        self.assertAlmostEqual(signed_rank([1, 2, 3, 4, 5])["p_value"], 2/32)

    def test_ties_zeros_and_insufficient_seeds(self):
        self.assertEqual(signed_rank([0]*5)["p_value"], 1)
        self.assertEqual(signed_rank([1, 1, 0, -1], min_seeds=2)["p_value"], 1)
        self.assertIsNone(signed_rank([1, 2])["p_value"])
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_duplicate_trials_rejected(self):
        config = json.loads((ROOT / "configs/demo_quick.json").read_text())
        r = {"method": "F4", "training_seed": 0, "condition": "nominal",
             "environment_seed": 201, "corruption_replicate": 11}
        with self.assertRaises(ValueError):
            check_matrix(pd.DataFrame([r, r]), config)


if __name__ == "__main__":
    unittest.main()
