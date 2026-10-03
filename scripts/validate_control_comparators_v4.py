import os,sys,json
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from iotexp.improvement_runtime import ImprovedBackend,V4
from iotexp.supplement_common import base_config
from iotexp.control_supplement import FlowMPC,MPCConfig
from iotexp.util import write_json
from iotexp.engine import evaluate_episode
from iotexp.telemetry import Condition
import run_control_supplement as runner

def main():
    torch.set_num_threads(1)
    for predictor in ['combined','graph_wavenet']:
        backend=ImprovedBackend(base_config(),predictor)
        backend.prime_causal_cache(np.array([126,132,138,144,148,152]))
        for day in [0,6,12,18,22,26]:
            folder=V4/'control_comparators'/predictor/f'day{day}'
            if (folder/'result.json').exists():continue
            r=runner.evaluate_control_episode(backend,'validation',day,MPCConfig(horizon=6,time_limit_seconds=1),folder/'trace.npz',folder/'solver.jsonl')
            write_json(folder/'result.json',r)
            print(predictor,day,r['restricted_mean_wait_min'],flush=True)

if __name__=='__main__':main()
