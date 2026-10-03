"""Frozen v2 inputs and declared v3 scenarios; never refit on evaluation data."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
from .scenic import ScenicBackend
from .scenic_forecast import load_forecaster
from .util import digest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'runs/scenic_rebuild_v2'
V3 = ROOT / 'runs/supplement_v3'
EXTERNAL_SEEDS = (20261001, 20261002, 20261003)


def base_config():
    return json.loads((V2 / 'config.json').read_text(encoding='utf-8'))


def verify_v2():
    manifest = json.loads((V2 / 'manifest.json').read_text(encoding='utf-8'))
    frozen = json.loads((V3 / 'frozen_v2_source/source_manifest.json').read_text(encoding='utf-8'))
    if digest(V2 / 'config.json') != manifest['config_sha256']:
        raise ValueError('v2 config changed')
    for name, expected in frozen['files'].items():
        if digest(ROOT / 'iotexp' / name) != expected:
            raise ValueError('Original source modified: ' + name)
    for relative, expected in manifest['asset_sha256'].items():
        if digest(V2 / relative) != expected:
            raise ValueError('Original asset modified: ' + relative)
    return manifest


class SupplementBackend(ScenicBackend):
    """Reuse frozen training statistics and predictor, with causal-input memoization.

    The cache key is the complete forecaster input (history values and timestamp).
    It is independent of future truth and never used for timing experiments.
    """
    def __init__(self, config, predictor='edge_stgru', cache_forecasts=True):
        super().__init__(config)
        self.stats, self.forecaster = load_forecaster(V2 / 'assets')
        if predictor == 'ridge':
            with np.load(V2 / 'assets/forecast/ridge_seed-1.npz', allow_pickle=False) as z:
                self.forecaster = {k: z[k].copy() for k in z.files if k != 'kind'}
            self.forecaster['kind'] = 'ridge'
            self.forecaster['config'] = json.loads((V2 / 'assets/forecast/configuration.json').read_text())
        elif predictor != 'edge_stgru':
            raise ValueError(predictor)
        self.predictor_name = predictor
        self.cache_forecasts = cache_forecasts
        self._forecast_cache = {}

    def forecast(self, window):
        if not self.cache_forecasts:
            return super().forecast(window)
        key = (int(window.decision_time), np.asarray(window.values[-self.history:, :, 0], dtype=np.float32).tobytes())
        if key not in self._forecast_cache:
            self._forecast_cache[key] = super().forecast(window)
        return self._forecast_cache[key].copy()

    def dataset_hashes(self):
        return {k: hashlib.sha256(np.asarray(self.data[k]).tobytes()).hexdigest()
                for k in ('inflow', 'demand', 'adjacency', 'edge_lengths_m')}


def load_base_backend(predictor='edge_stgru', cache_forecasts=True):
    return SupplementBackend(base_config(), predictor, cache_forecasts)


def make_scenario_backend(seed, surge=False, predictor='edge_stgru', cache_forecasts=True):
    config = deepcopy(base_config())
    config['scenic']['dataset_seed'] = int(seed)
    if surge:
        for key in ('gate_base', 'gate_peak'):
            config['scenic'][key] = [1.6 * value for value in config['scenic'][key]]
        config['scenic']['event_amplitude'] *= 1.6
    return SupplementBackend(config, predictor, cache_forecasts)


def scenarios():
    return [('legacy', 20260929, False)] + [
        (f'seed{seed}_{"surge" if surge else "nominal"}', seed, surge)
        for surge in (False, True) for seed in EXTERNAL_SEEDS]

