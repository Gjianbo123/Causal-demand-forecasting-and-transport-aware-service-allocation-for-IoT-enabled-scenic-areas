import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from iotexp.scenic import scenic_config, scenic_graph, build_dataset, ScenicEnvironment, ScenicBackend, repair_target
from iotexp.telemetry import TrainingStats, TelemetryGateway, Condition


class ScenicRebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = scenic_config({})
        cls.data = build_dataset(cls.cfg)

    def test_graph_dataset_and_day_partition_are_deterministic(self):
        graph = scenic_graph()
        self.assertEqual(graph['edge_index'].shape,(31,2))
        self.assertTrue(np.isfinite(graph['shortest_distance_m']).all())
        self.assertEqual(self.data['inflow'].shape,(180,87,24))
        np.testing.assert_array_equal(self.data['inflow'],build_dataset(self.cfg)['inflow'])
        days = [set(ScenicEnvironment(self.cfg,self.data,s,i).day for i in range(n))
                for s,n in zip(('train','validation','test'),(126,27,27))]
        self.assertEqual(len(set.union(*days)),180)
        self.assertFalse(days[0]&days[1] or days[0]&days[2] or days[1]&days[2])

    def test_resource_request_conservation_and_terminal_backlog(self):
        env = ScenicEnvironment(self.cfg,self.data,'test',0)
        env.reset()
        rng = np.random.default_rng(43)
        arrivals = served = 0
        for t in range(72):
            frame,reward,done,audit = env.step(rng.normal(0,1,(4,24)))
            arrivals += audit.arrivals_count; served += len(audit.served_waits_min)
            np.testing.assert_array_equal((env.inventory+env.reserved).sum(1),env.totals)
            self.assertTrue(((env.inventory+env.reserved)<=env.holding).all())
            self.assertFalse(audit.executable_infeasible)
            self.assertEqual(arrivals,served+audit.unserved_count)
            self.assertEqual(done,t==71)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros((4,24)))

    def test_transit_is_unavailable_and_committed_targets_cannot_be_recalled(self):
        env = ScenicEnvironment(self.cfg,self.data,'train',0)
        env.reset()
        raw = np.zeros((4,24)); raw[:,-9] = 20
        _,_,_,audit = env.step(raw)
        self.assertGreater(env.reserved.sum(),0)
        self.assertGreater(sum(x[3] for x in env.trips),0)
        np.testing.assert_array_equal(env.reserved.sum(1),
                                      env.totals-env.inventory.sum(1))
        reserved = env.reserved.copy()
        reversed_raw = -raw
        _,target,_ = repair_target(reversed_raw,env.totals,env.holding,reserved)
        self.assertTrue((target>=reserved).all())

    def test_known_fifo_waits_and_independent_request_counts(self):
        cfg = copy.deepcopy(self.cfg)
        data = dict(self.data)
        data['demand'] = np.zeros_like(data['demand'])
        data['demand'][0,15,0,0] = 6  # t=1; 2 units at node0 -> 4 requests/interval
        env = ScenicEnvironment(cfg,data,'train',0)
        env.reset()
        # Zero scores reproduce the initial uniform allocation and cause no trips.
        _,_,_,a = env.step(np.zeros((4,24)))
        np.testing.assert_array_equal(a.served_waits_min,[0,0,0,0])
        self.assertEqual(a.unserved_count,2)
        self.assertEqual(a.queue_area_min,0)
        _,_,_,b = env.step(np.zeros((4,24)))
        np.testing.assert_array_equal(b.served_waits_min,[10,10])
        self.assertEqual(b.unserved_count,0)
        self.assertEqual(b.queue_area_min,20)
        self.assertEqual(env.arrived_requests,6)

    def test_telemetry_contains_only_available_observations(self):
        env = ScenicEnvironment(self.cfg,self.data,'test',0)
        samples = self.data['observations'][:126].reshape(-1,24,5)
        times = np.tile(self.data['times'],126)
        stats = TrainingStats.fit(samples,times,72)
        gateway = TelemetryGateway(stats,Condition(delay=2),9,12)
        for frame in env.reset():
            gateway.submit(frame.time,frame.visitor)
        window = gateway.window(0)
        self.assertLessEqual(np.nanmax(window.source_receipt),0)
        self.assertLessEqual(np.nanmax(window.source_event),-2)
        snapshot = window.values.copy()
        frame,_,_,_ = env.step(np.zeros((4,24)))
        gateway.submit(frame.time,frame.visitor)
        np.testing.assert_array_equal(snapshot,gateway.window(0).values)

    def test_control_forecast_uses_public_closing_time_only(self):
        # Construct a backend without a dataset: masking must not read future truth.
        backend = object.__new__(ScenicBackend)
        backend.config, backend.cfg = {}, {"episode_steps":72}
        backend.horizons = [1,2,4,6]
        backend.forecaster, backend.stats = None, None
        with patch("iotexp.scenic_forecast.forecast_window", side_effect=lambda *args: np.full((24,4),7,dtype=np.float32)):
            early = backend.forecast(SimpleNamespace(decision_time=66))
            np.testing.assert_array_equal(early,np.full((24,4),7))
            later = backend.forecast(SimpleNamespace(decision_time=68))
            np.testing.assert_array_equal(later,np.tile([7,7,7,0],(24,1)))
            last = backend.forecast(SimpleNamespace(decision_time=71))
            np.testing.assert_array_equal(last,np.tile([7,0,0,0],(24,1)))


if __name__ == '__main__':
    unittest.main()
