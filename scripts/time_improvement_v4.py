"""Serial uncached timing on validation day126 after evaluation workers stop."""
import os,sys,json,time,platform
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from iotexp.improvement_runtime import ImprovedBackend,V4
from iotexp.forecast_improvement import Anchor
from iotexp.control_supplement import FlowMPC,MPCConfig
from iotexp.transport_rollout import TransportRollout
from iotexp.improvement_controllers import SeasonalMPC
from iotexp.supplement_common import base_config
from iotexp.util import write_json
from run_improvement_final import verify
import run_control_supplement as runner

def main():
    verify();torch.set_num_threads(1);anchor=Anchor.load(V4/'forecast/anchor.npz');rows=[]
    for method,h in [('rollout',12),('combined_mpc6',6),('combined_mpc12',12)]:
        path=V4/'serial_timing'/method
        if (path/'result.json').exists():rows.append(json.loads((path/'result.json').read_text()));continue
        backend=ImprovedBackend(base_config(),'combined',cache=False)
        if method=='rollout':runner.FlowMPC=lambda c,d,o:TransportRollout(c,d,o,anchor,0)
        elif method=='combined_mpc12':runner.FlowMPC=lambda c,d,o:SeasonalMPC(c,d,o,anchor,0)
        else:runner.FlowMPC=FlowMPC
        r=runner.evaluate_control_episode(backend,'validation',0,MPCConfig(horizon=h,time_limit_seconds=1),path/'trace.npz',path/'solver.jsonl')
        with np.load(path/'trace.npz') as z:
            latency=z['local_pipeline_ms'][5:]
            r.update(method=method,median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),measured_decisions=len(latency))
        write_json(path/'result.json',r);rows.append(r);print(method,r['median_ms'],r['p95_ms'],flush=True)
    write_json(V4/'serial_timing/summary.json',dict(results=rows,day=126,conditions='serial, uncached,CPU one PyTorch thread; first5 decisions discarded; warm model but varying queue states',
        included='causal history preparation, forecast ensemble, planning and executable-flow construction',
        excluded='model loading, physical network and actuation, service-simulator transition; not end-to-end field latency',platform=platform.platform()))

if __name__=='__main__':main()
