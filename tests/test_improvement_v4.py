import unittest
from types import SimpleNamespace
import numpy as np
from iotexp.transport_rollout import TransportRollout
from iotexp.control_supplement import MPCConfig
from iotexp.scenic import DEFAULT_SCENIC

class ImprovementTests(unittest.TestCase):
    def tiny(self):
        cfg=dict(DEFAULT_SCENIC)
        cfg.update(resource_totals=[1],holding_per_service_node=[1],service_per_resource=[1],base_service=[0],
                   request_probability=[1],resource_speed_m_per_min=[100],resource_setup_minutes=[0],
                   movement_penalty=.1,queue_reward_scale=1,overload_penalty=0,queue_capacity_per_service_node=100)
        return TransportRollout(cfg,np.array([[0,100],[100,0.]]),MPCConfig(horizon=3),None)

    def test_queue_reflection_matches_stepwise_service(self):
        rng=np.random.default_rng(7);q0=rng.integers(0,10,(4,24));net=rng.normal(0,4,(18,4,24))
        actual=[];q=q0.copy()
        for delta in net:q=np.maximum(q+delta,0);actual.append(q.copy())
        np.testing.assert_allclose(TransportRollout.queues(q0,net),actual,atol=1e-12)

    def test_dispatch_loses_one_interval_of_service(self):
        c=self.tiny();d=np.array([[[0,2]],[[0,2]],[[0,2.]]])
        r=c.solve(np.array([[1,0]]),np.zeros((1,2)),d,np.zeros_like(d),np.zeros_like(d))
        self.assertEqual(r['transfers'][0,0,1],1)
        np.testing.assert_array_equal(r['predicted_queue'][:,0,1],[2,3,4])

    def test_committed_arrival_not_recalled(self):
        c=self.tiny();d=np.array([[[0,2]],[[0,2]],[[0,2.]]]);a=np.zeros_like(d);a[1,0,1]=1
        p=np.zeros_like(d);p[:2,0,1]=1
        r=c.solve(np.zeros((1,2),int),np.zeros((1,2)),d,a,p)
        self.assertEqual(r['transfers'].sum(),0)
        np.testing.assert_array_equal(r['predicted_queue'][:,0,1],[2,4,5])

if __name__=='__main__':unittest.main()
