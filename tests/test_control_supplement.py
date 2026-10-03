import copy
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from iotexp.control_supplement import FlowMPC, MPCConfig, execute_dispatch, target_scores, validate_transfers
from iotexp.scenic import DEFAULT_SCENIC, ScenicEnvironment, build_dataset, scenic_config


def tiny_config():
    cfg = dict(DEFAULT_SCENIC)
    cfg.update(resource_totals=[1], holding_per_service_node=[1], service_per_resource=[1],
               base_service=[0], request_probability=[1], resource_speed_m_per_min=[100],
               resource_setup_minutes=[0], movement_penalty=.1, queue_reward_scale=1,
               overload_penalty=0, queue_capacity_per_service_node=100)
    return cfg


class ControlSupplementTests(unittest.TestCase):
    def test_integer_mpc_matches_exhaustive_tiny_dynamic_program(self):
        cfg=tiny_config(); distance=np.array([[0,100],[100,0.]])
        control=FlowMPC(cfg,distance,MPCConfig(time_limit_seconds=2,relative_gap=0))
        demand=np.array([[[0,2]],[[0,2]],[[0,2]]],dtype=float)
        got=control.solve(np.array([[1,0]]),np.zeros((1,2)),demand,
                          np.zeros((3,1,2)),np.zeros((3,1,2)))
        # One resource; every route is an exact stay or one-interval move.
        best=np.inf
        for moves in itertools.product([False,True],repeat=3):
            position=0; queue=np.zeros(2); objective=0
            for t,move in enumerate(moves):
                queue+=demand[t,0]
                if move:
                    position=1-position; objective+=.01
                else:
                    queue[position]=max(0,queue[position]-1)
                objective+=queue.sum()/2
            best=min(best,objective)
        self.assertTrue(got["accepted"])
        self.assertAlmostEqual(got["objective"],best,places=7)
        self.assertEqual(got["transfers"][0,0,1],1)
        self.assertEqual(got["transfers"].sum(),1)

    def test_existing_trip_is_not_recalled_or_served_before_arrival(self):
        control=FlowMPC(tiny_config(),np.array([[0,100],[100,0.]]),MPCConfig(time_limit_seconds=2,relative_gap=0))
        demand=np.array([[[0,2]],[[0,2]],[[0,2]]],dtype=float)
        arrivals=np.zeros_like(demand); arrivals[1,0,1]=1
        pending=np.zeros_like(demand); pending[:2,0,1]=1
        got=control.solve(np.zeros((1,2)),np.zeros((1,2)),demand,arrivals,pending)
        self.assertTrue(got["accepted"])
        self.assertEqual(got["transfers"].sum(),0)
        self.assertAlmostEqual(got["objective"],(2+4+5)/2)

    def test_no_incumbent_means_feasible_hold(self):
        control=FlowMPC(tiny_config(),np.array([[0,100],[100,0.]]))
        with patch("iotexp.control_supplement.milp",return_value=SimpleNamespace(status=1,x=None,message="time limit")):
            got=control.solve(np.array([[1,0]]),np.zeros((1,2)),np.zeros((3,1,2)),
                              np.zeros((3,1,2)),np.zeros((3,1,2)))
        self.assertTrue(got["fallback"])
        self.assertEqual(got["transfers"].sum(),0)

    def test_fractional_expected_requests_remain_feasible(self):
        control=FlowMPC(tiny_config(),np.array([[0,100],[100,0.]]),MPCConfig(time_limit_seconds=2,relative_gap=0))
        demand=np.array([[[.35,1.75]],[[.15,2.55]],[[.4,2.25]]])
        got=control.solve(np.array([[1,0]]),np.array([[.1,.25]]),demand,
                          np.zeros_like(demand),np.zeros_like(demand))
        self.assertTrue(got["accepted"])
        self.assertLess(got["primal_residual"],1e-6)
        best=np.inf
        for moves in itertools.product([False,True],repeat=3):
            position=0; queue=np.array([.1,.25]); objective=0
            for t,move in enumerate(moves):
                queue+=demand[t,0]
                if move: position=1-position; objective+=.01
                else: queue[position]=max(0,queue[position]-1)
                objective+=queue.sum()/2
            best=min(best,objective)
        self.assertAlmostEqual(got["objective"],best,places=7)

    def test_reserved_capacity_cannot_be_overbooked(self):
        with self.assertRaises(ValueError):
            validate_transfers(np.array([[[0,1],[0,0]]]),np.array([[1,0]]),
                               np.array([[0,1]]),np.array([[1,1]]))

    def test_executor_matches_original_service_and_trip_timing(self):
        cfg=scenic_config({"days":3,"split_days":[1,1,1]})
        env=ScenicEnvironment(cfg,build_dataset(cfg),"test",0); env.reset()
        original=copy.deepcopy(env)
        flow=np.zeros((4,24,24),dtype=int); flow[0,0,1]=1; flow[3,0,15]=1
        target=original.inventory+original.reserved-flow.sum(2)+flow.sum(1)
        first=original.step(target_scores(target))
        second=execute_dispatch(env,flow)
        self.assertAlmostEqual(first[1],second[1])
        self.assertAlmostEqual(first[3].movement_cost,second[3].movement_cost)
        for _ in range(5):
            np.testing.assert_array_equal(original.inventory,env.inventory)
            np.testing.assert_array_equal(original.reserved,env.reserved)
            np.testing.assert_array_equal(original.queue_batches,env.queue_batches)
            self.assertEqual(original.trips,env.trips)
            first=original.step(target_scores(original.inventory+original.reserved))
            second=execute_dispatch(env,np.zeros_like(flow))
            np.testing.assert_array_equal(first[0].visitor,second[0].visitor)
            np.testing.assert_array_equal(first[3].served_waits_min,second[3].served_waits_min)
            self.assertAlmostEqual(first[1],second[1])

    def test_causal_api_and_commitment_ledger(self):
        control=FlowMPC(tiny_config(),np.array([[0,100],[100,0.]]),MPCConfig(time_limit_seconds=2,relative_gap=0))
        observed=np.zeros((1,2,2))
        window=SimpleNamespace(values=observed,decision_time=0)
        ledger=np.r_[1,0,0,0,0,0]
        pred=np.array([[0,0,0,0],[2,2,2,2.]])
        first=control.decide(window,ledger,pred,[1,2,4,6])
        # The optimizer has no backend, data, day index or future-truth parameter.
        second=control.decide(window,ledger.copy(),pred.copy(),[1,2,4,6])
        np.testing.assert_array_equal(first["transfers"],second["transfers"])
        with self.assertRaises(ValueError):
            control.resource_snapshot(np.r_[0,0,0,1,0,2])


if __name__=="__main__":
    unittest.main()
