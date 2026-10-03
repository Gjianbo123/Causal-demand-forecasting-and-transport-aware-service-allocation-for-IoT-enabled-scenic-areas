"""Finite controller search on predeclared original development days only."""
import os,sys,json,copy,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from iotexp.improvement_runtime import ImprovedBackend,ForecastRuntime,V4
from iotexp.forecast_improvement import Anchor
from iotexp.control_supplement import FlowMPC,MPCConfig
from iotexp.transport_rollout import TransportRollout
from iotexp.supplement_common import base_config,verify_v2
from iotexp.util import write_json,digest
sys.path.insert(0,str(ROOT/'scripts'))
import run_control_supplement as runner

def main():
    verify_v2();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=V4/'control_development';out.mkdir(exist_ok=True)
    protocol=json.loads((V4/'experiment_plan.json').read_text())
    anchor=Anchor.load(V4/'forecast/anchor.npz');backend=ImprovedBackend(base_config())
    backend.prime_causal_cache(np.array([126,132,138,144,148,152]))
    candidates=[(h,t) for h in [6,12,18] for t in [0,4]];results=[]
    for h,t in candidates:
        runner.FlowMPC=lambda cfg,distance,options:TransportRollout(cfg,distance,options,anchor,t)
        rows=[]
        for day in [0,6,12]:
            folder=out/f'H{h}_T{t}'/f'day{day}'
            if (folder/'result.json').exists():r=json.loads((folder/'result.json').read_text())
            else:
                r=runner.evaluate_control_episode(backend,'validation',day,MPCConfig(horizon=h),folder/'trace.npz',folder/'solver.jsonl')
                write_json(folder/'result.json',r)
            rows.append(r)
        score=sum(r['waiting_area_min'] for r in rows)/sum(r['arrivals_count'] for r in rows)
        results.append(dict(horizon=h,terminal_weight=t,development_wait=score));print(results[-1],flush=True)
    selected=min(results,key=lambda r:r['development_wait'])
    write_json(out/'selection.json',dict(selected=selected,candidates=results,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p:digest(ROOT/p) for p in ['iotexp/transport_rollout.py','iotexp/improvement_runtime.py','scripts/validate_transport_rollout.py']},
        plan_sha256=digest(V4/'experiment_plan.json')))
    h,t=selected['horizon'],selected['terminal_weight']
    runner.FlowMPC=lambda cfg,distance,options:TransportRollout(cfg,distance,options,anchor,t)
    rows=[]
    for day in [18,22,26]:
        folder=out/'confirmation'/f'day{day}'
        r=runner.evaluate_control_episode(backend,'validation',day,MPCConfig(horizon=h),folder/'trace.npz',folder/'solver.jsonl')
        write_json(folder/'result.json',r);rows.append(r)
    write_json(out/'confirmation.json',dict(wait=sum(r['waiting_area_min'] for r in rows)/sum(r['arrivals_count'] for r in rows),episodes=rows))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
