"""Serial, uncached CPU decision timing, run after the training/evaluation workers."""
from pathlib import Path
import csv
import json
import os
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from iotexp.engine import Session,make_policy,tensor
from iotexp.telemetry import Condition
from iotexp.control_supplement import FlowMPC,MPCConfig,execute_dispatch
from iotexp.supplement_common import V3,load_base_backend,verify_v2,base_config
from iotexp.util import digest,write_json,environment_info


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);verify_v2()
    out=V3/'serial_latency';out.mkdir(parents=True,exist_ok=True)
    target=out/'decisions.csv'
    if target.exists():raise FileExistsError(target)
    solver=json.loads((V3/'control/solver_protocol.json').read_text())['options']
    options=MPCConfig(**solver)
    configurations=[('static','edge_stgru'),('reactive','edge_stgru'),('lookahead','edge_stgru'),
                    ('F0','edge_stgru'),('F1','edge_stgru'),('F4','edge_stgru'),('F4','ridge'),('MPC','edge_stgru')]
    protocol=dict(days=[0,13,26],split='original validation',training_seed_for_policies=0,
                  workers=1,torch_threads=1,cached_forecasts=False,
                  warmup='Discard first five decisions of each 72-step daily rollout; all runtime imports/checkpoint loading are outside measurement',
                  measurements_per_method=201,scope='Sum of instrumented causal window/state, actual required prediction, controller, and dispatch/repair stages, including MPC acknowledged-transfer bookkeeping; excludes physical communications, service simulation, logging and checkpoint load',
                  f0='Skip unused diagnostic shadow forecast; actor has no forecast inputs',
                  hardware=environment_info(),solver_options=solver,
                  source_sha256=digest(Path(__file__)))
    write_json(out/'protocol.json',protocol)
    rows=[]
    for method,predictor in configurations:
        backend=load_base_backend(predictor,cache_forecasts=False)
        cfg=base_config();model=None;checkpoint_hash=None
        if method.startswith('F'):
            checkpoint=V3/'policies'/predictor/'checkpoints'/f'{method}_seed0'/'best.pt'
            model=make_policy(backend,cfg,method,0)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']);model.eval()
            checkpoint_hash=digest(checkpoint)
        for day in protocol['days']:
            session=Session(backend,'validation',day,Condition(),11)
            controller=FlowMPC(backend.cfg,backend.data['shortest_distance_m'],options) if method=='MPC' else None
            for step in range(72):
                began=time.perf_counter()
                window=session.gateway.window(session.frame.time)
                state=backend.encode_state(window,session.frame.ledger.copy()) if model is not None else None
                state_ms=1000*(time.perf_counter()-began)
                before_forecast=time.perf_counter()
                if method in ('static','reactive','F0'):
                    forecast=None
                else:
                    forecast=backend.forecast(window)
                forecast_ms=1000*(time.perf_counter()-before_forecast)
                before_decision=time.perf_counter()
                if model is not None:
                    values=np.zeros((backend.nodes,len(backend.horizons)),dtype=np.float32) if method=='F0' else (forecast-backend.stats.mean[:,0,None])/backend.stats.scale[:,0,None]
                    raw=model.act(tensor(state[None],'cpu'),tensor(values[None],'cpu'),deterministic=True)[0][0].numpy()
                elif controller is not None:
                    decision=controller.decide(window,session.frame.ledger.copy(),forecast,backend.horizons)
                elif method=='lookahead':
                    observed=np.maximum(window.values[-1],0);queue=observed[:,1:].T
                    distance=backend.data['shortest_distance_m'][:16,:16]
                    positive=distance[distance>0]
                    leads=np.ceil((np.median(positive)/np.asarray(backend.cfg['resource_speed_m_per_min'])+np.asarray(backend.cfg['resource_setup_minutes']))*backend.cfg['travel_time_multiplier']/backend.cfg['decision_minutes'])
                    indices=[int(np.argmin(np.abs(np.asarray(backend.horizons)-lead))) for lead in leads]
                    expected=np.asarray(backend.cfg['request_probability'])[:,None]*np.stack([forecast[:,j] for j in indices])
                    raw=np.log(np.maximum((queue+expected)/np.asarray(backend.cfg['service_per_resource'])[:,None]+.1,1e-6)).astype(np.float32)
                else:raw=backend.heuristic_action(window,method)
                decision_ms=1000*(time.perf_counter()-before_decision)
                ledger_update_ms=0.0
                if controller is None:
                    _,done,_=session.step(raw)
                else:
                    frame,_,done,_=execute_dispatch(session.env,decision['transfers'])
                    before_ledger=time.perf_counter()
                    controller.after_step(decision['transfers'])
                    ledger_update_ms=1000*(time.perf_counter()-before_ledger)
                    session.frame=frame;session.gateway.submit(frame.time,frame.visitor)
                repair_ms=float(session.env.last_repair_ms)+ledger_update_ms
                rows.append(dict(method=method,predictor=predictor,day=day,step=step,warmup=step<5,
                                 state_ms=state_ms,forecast_ms=forecast_ms,decision_ms=decision_ms,
                                 repair_ms=repair_ms,ledger_update_ms=ledger_update_ms,
                                 pipeline_ms=state_ms+forecast_ms+decision_ms+repair_ms,
                                 checkpoint_sha256=checkpoint_hash))
            print('Serial timing',method,predictor,'day',day,flush=True)
    with target.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import pandas as pd
    data=pd.DataFrame(rows);data=data[~data.warmup]
    results=[]
    for (method,predictor),g in data.groupby(['method','predictor']):
        r=dict(method=method,predictor=predictor,n=len(g))
        assert len(g)==201
        for key in ('state_ms','forecast_ms','decision_ms','repair_ms','pipeline_ms'):
            r[key+'_mean']=float(g[key].mean());r[key+'_median']=float(g[key].median());r[key+'_p95']=float(g[key].quantile(.95))
        results.append(r)
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
