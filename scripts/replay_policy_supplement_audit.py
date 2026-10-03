"""Prespecified seed-zero/day-zero causal/action replay audit, not a new trial.

Replays F0/F1/F4 on each of the three new nominal and three surge worlds: 18
episodes in total. Verifies every non-timing episode metric against the stored
long-budget evaluation shard. It does not change checkpoints or select models.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from iotexp.engine import evaluate_episode,make_policy
from iotexp.supplement_common import V3,make_scenario_backend,scenarios,verify_v2
from iotexp.telemetry import Condition
from iotexp.util import digest,write_json


def main():
    start=time.perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_v2()
    out=V3/"audits/full_policy_replay"
    if out.exists():raise FileExistsError("Refusing to overwrite the bounded replay audit")
    out.mkdir(parents=True)
    directory=V3/"policies/edge_stgru"
    config=json.loads((directory/"config.json").read_text())
    config["trace_mode"]="full"
    scenarios_to_check=[s for s in scenarios() if s[0]!="legacy"]
    protocol=dict(scope="seed0 and day offset0 only; not all-policy or all-day action replay",
                  methods=["F0","F1","F4"],predictor="edge_stgru",training_seeds=[0],
                  environment_seeds=[0],scenario_names=[s[0] for s in scenarios_to_check],
                  expected_episodes=18,checkpoint_budget_steps=43200,corruption_replicate=11,
                  trace_mode="full",metric_absolute_tolerance=1e-6,metric_relative_tolerance=1e-7,
                  purpose="evidence audit; no training, tuning, model selection or inferential replication",
                  shared_protocol_sha256=digest(V3/"protocol.json"),script_sha256=digest(Path(__file__)))
    write_json(out/"protocol.json",protocol)
    records=[];metric_checks=0
    for method in protocol["methods"]:
        cp=directory/"checkpoints"/f"{method}_seed0"/"best.pt"
        info=json.loads((cp.parent/"metadata.json").read_text())
        assert digest(cp)==info["checkpoint_sha256"]
        shard=directory/"evaluation_shards"/f"{method}_seed0_long.csv"
        table=pd.read_csv(shard)
        for name,seed,surge in scenarios_to_check:
            backend=make_scenario_backend(seed,surge,cache_forecasts=False)
            model=make_policy(backend,config,method,0)
            state=torch.load(cp,map_location="cpu",weights_only=True)
            assert state["method"]==method and state["training_seed"]==0
            model.load_state_dict(state["state_dict"])
            selected=table[(table.scenario==name)&(table.environment_seed==0)]
            assert len(selected)==1
            original=selected.iloc[0]
            assert original.checkpoint_sha256==digest(cp)
            trace_path=out/f"{method}_{name}_day0_full.npz"
            corruption=int(np.random.SeedSequence([0,11,500009]).generate_state(1)[0])
            result=evaluate_episode(model,backend,config,"test",0,Condition(),corruption,trace_path)
            compared=[]
            for key,value in result.items():
                if "latency" in key:continue
                if key not in original.index:raise AssertionError(f"Missing stored metric {key}")
                if not np.isclose(value,original[key],rtol=1e-7,atol=1e-6,equal_nan=True):
                    raise AssertionError(f"Replay changed {method} {name} {key}: {value} != {original[key]}")
                compared.append(key);metric_checks+=1
            with np.load(trace_path,allow_pickle=False) as z:
                assert np.array_equal(z["decision_time"],np.arange(72))
                target=z["executed_action"]
                assert np.array_equal(target.sum(2),np.tile([30,24,16,12],(72,1)))
                assert (target[:,:,16:]==0).all()
                assert (target[:,:,:16]<=np.asarray([5,4,3,3])[None,:,None]).all()
                reserved=z["ledger"][:,96:192].reshape(72,4,24)
                assert (target>=reserved).all()
                receipt=z["source_receipt"]
                assert (np.isnan(receipt)|(receipt<=np.arange(72)[:,None,None])).all()
                event=z["source_event"]
                grid=np.arange(72)[:,None,None]-np.arange(11,-1,-1)[None,:,None]
                assert (np.isnan(event)|(event<=grid)).all()
                close_mask=np.arange(72)[:,None]+np.asarray([1,2,4,6])[None]>72
                assert (z["forecast"].transpose(0,2,1)[close_mask]==0).all()
                assert not z["executable_infeasible"].any()
                area=float(10*z["truth_visitor_before"][...,1:].sum())
                assert np.isclose(area,result["waiting_area_min"],atol=1e-6)
                # Full and original compact wait samples should be identical too.
                with np.load(ROOT/original.trace_path) as saved:
                    assert np.array_equal(z["served_waits_min"],saved["served_waits_min"])
            records.append(dict(method=method,scenario=name,dataset_seed=seed,surge=surge,
                                training_seed=0,environment_seed=0,checkpoint_sha256=digest(cp),
                                dataset_hashes=backend.dataset_hashes(),source_shard_sha256=digest(shard),
                                source_trace_sha256=digest(ROOT/original.trace_path),
                                full_trace=trace_path.relative_to(ROOT).as_posix(),full_trace_sha256=digest(trace_path),
                                compared_metrics=compared,metrics=result))
            print(f"Replay verified {method} {name} day0",flush=True)
    assert len(records)==18
    manifest=dict(status="PASS",**protocol,episodes=18,decision_epochs=1296,
                  non_timing_metric_comparisons=metric_checks,elapsed_seconds=time.perf_counter()-start,
                  checks=["frozen checkpoint identity","all non-timing metrics match stored evaluation",
                          "raw served-wait arrays identical","full observed queue-area reconstruction",
                          "resource target budgets and immutable arrival reservations","holding capacities",
                          "causal source event and receipt timestamps","public closing forecast mask",
                          "zero executable resource constraint flags"],records=records)
    write_json(out/"manifest.json",manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!="records"},indent=2))


if __name__=="__main__":main()
