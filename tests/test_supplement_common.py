import unittest
import numpy as np
from iotexp.supplement_common import load_base_backend, make_scenario_backend, verify_v2
from iotexp.engine import Session
from iotexp.telemetry import Condition


class SupplementCommonTests(unittest.TestCase):
    def test_original_source_and_assets_unchanged(self):
        verify_v2()

    def test_cached_forecast_matches_uncached_and_does_not_alias(self):
        b=load_base_backend()
        s=Session(b,'validation',0,Condition(),11)
        w=s.gateway.window(0)
        cached=b.forecast(w)
        b.cache_forecasts=False
        np.testing.assert_array_equal(cached,b.forecast(w))
        b.cache_forecasts=True
        cached[:]=999
        self.assertFalse(np.all(b.forecast(w)==999))
        # Same time with different causal values must not reuse the cached prediction.
        w.values[:,:,0] += 4
        new=b.forecast(w)
        b.cache_forecasts=False
        np.testing.assert_array_equal(new,b.forecast(w))

    def test_external_scenario_keeps_training_stats_and_changes_demand(self):
        a=load_base_backend();b=make_scenario_backend(20261001,True)
        np.testing.assert_array_equal(a.stats.mean,b.stats.mean)
        np.testing.assert_array_equal(a.stats.scale,b.stats.scale)
        self.assertNotEqual(a.dataset_hashes()['inflow'],b.dataset_hashes()['inflow'])
        self.assertEqual(b.cfg['event_amplitude'],1.6*a.cfg['event_amplitude'])

    def test_ridge_is_loaded_without_refitting(self):
        b=load_base_backend('ridge')
        s=Session(b,'validation',0,Condition(),11)
        result=b.forecast(s.gateway.window(0))
        self.assertEqual(result.shape,(24,4))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result>=0).all())


if __name__=='__main__': unittest.main()
