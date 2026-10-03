"""Supplementary flow-MPC runs; validation precedes a frozen formal protocol.

No files in scenic_rebuild_v2 are edited. Formal datasets/identities are supplied
by the supplement's shared protocol. Every solver return and executed flow is
retained, including failed solves and unfavorable performance.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import numpy as np
import torch
from iotexp.control_supplement import FlowMPC,MPCConfig,execute_dispatch
from iotexp.engine import Session
from iotexp.metrics import Metrics
from iotexp.scenic import ScenicBackend
from iotexp.telemetry import Condition
from iotexp.util import digest,write_json


def load_original_backend(base):
    cfg=json.loads((base/"config.json").read_text(encoding="utf-8"))
    backend=ScenicBackend(cfg); backend.load(base/"assets")
    return backend


def evaluate_control_episode(backend,split,day,options,trace_path,solver_path,oracle=False):
    corruption=int(np.random.SeedSequence([day,11,500009]).generate_state(1)[0])
    session=Session(backend,split,day,Condition(),corruption)
    control=FlowMPC(backend.cfg,backend.data["shortest_distance_m"],options)
    metric=Metrics(); area=0.; trace={}; statuses=Counter(); fallbacks=0
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    solver_path.parent.mkdir(parents=True,exist_ok=True)
    if trace_path.exists() or solver_path.exists():
        raise FileExistsError("Refusing to overwrite an MPC episode")
    def append(key,value): trace.setdefault(key,[]).append(np.asarray(value).copy())
    def padded(value):
        result=np.zeros((options.horizon,*value.shape[1:]),dtype=value.dtype)
        result[:len(value)]=value
        return result
    with solver_path.open("x",encoding="utf-8") as log:
        for step in range(backend.cfg["episode_steps"]):
            started=time.perf_counter()
            window=session.gateway.window(session.frame.time)
            forecast=backend.forecast(window)
            ledger=session.frame.ledger.copy()
            if oracle:
                # Explicit perfect-information diagnostic ONLY. Not an online comparator.
                inv,_=control.resource_snapshot(ledger)
                horizon=min(options.horizon,backend.cfg["episode_steps"]-step)
                start=session.env.warmup+step+1
                demand=backend.data["demand"][session.env.day,start:start+horizon].copy()
                arrivals=np.zeros_like(demand,dtype=float); pending=np.zeros_like(arrivals)
                for k,_,j,amount,eta in control.commitments:
                    if eta<=horizon: arrivals[eta-1,k,j]+=amount
                    pending[:min(eta,horizon),k,j]+=amount
                decision=control.solve(inv,np.maximum(window.values[-1,:,1:].T,0),demand,arrivals,pending)
                decision["forecast_requests"]=demand
            else:
                decision=control.decide(window,ledger,forecast,backend.horizons)
            logic_ms=1000*(time.perf_counter()-started)
            report={key:value for key,value in decision.items() if not isinstance(value,np.ndarray)}
            report.update(decision_time=step,oracle_future_requests=bool(oracle),
                          controller_logic_ms=logic_ms,dispatch_units=int(decision["transfers"].sum()))
            statuses[str(report["status"])]+=1; fallbacks+=int(report["fallback"])
            append("decision_time",step); append("ledger_before",ledger)
            append("observed",window.values); append("mask",window.mask); append("age",window.age)
            append("source_event",window.source_event); append("source_receipt",window.source_receipt)
            append("forecast",forecast)
            for key in ("plan_transfers","predicted_queue","forecast_requests"):
                append(key,padded(decision[key]))
            append("transfers",decision["transfers"])
            frame,reward,done,audit=execute_dispatch(session.env,decision["transfers"])
            control.after_step(decision["transfers"])
            session.frame=frame; session.gateway.submit(frame.time,frame.visitor)
            control.resource_snapshot(frame.ledger)  # commits must match causal acknowledgement
            metric.add(audit); area+=audit.queue_area_min
            for key,value in dict(ledger_after=frame.ledger,reward=reward,queue_area_min=audit.queue_area_min,
                occupancy=audit.occupancy,unserved_count=audit.unserved_count,arrivals_count=audit.arrivals_count,
                movement_cost=audit.movement_cost,execution_infeasible=audit.executable_infeasible,
                controller_logic_ms=logic_ms,optimization_ms=decision["optimization_ms"],
                local_pipeline_ms=logic_ms+session.env.last_repair_ms).items(): append(key,value)
            report["local_pipeline_ms"]=logic_ms+session.env.last_repair_ms
            log.write(json.dumps(report,allow_nan=False)+"\n"); log.flush()
            if done: break
    result=metric.result()
    censored=float((session.env.queue_batches*(session.env.t-np.arange(session.env.queue_batches.shape[-1]))).sum()*backend.cfg["decision_minutes"])
    if not np.isclose(area,result["wait_sum_min"]+censored):
        raise RuntimeError("Completed and censored waiting identity failed")
    if result["served_count"]+result["unserved_count"]!=result["arrivals_count"]:
        raise RuntimeError("Request conservation failed")
    trace["served_waits_min"]=np.asarray(metric.waits,dtype=np.float32)
    trace["terminal_queue_batches"]=session.env.queue_batches.copy()
    np.savez_compressed(trace_path,**{key:np.asarray(value) for key,value in trace.items()})
    times=np.asarray(trace["optimization_ms"])
    result.update(waiting_area_min=area,restricted_mean_wait_min=area/max(1,result["arrivals_count"]),
        unserved_pct=100*result["unserved_count"]/max(1,result["arrivals_count"]),
        mean_reward=float(np.mean(trace["reward"])),solver_fallback_steps=fallbacks,
        solver_status_counts=json.dumps(dict(statuses),sort_keys=True),
        optimization_mean_ms=float(times.mean()),optimization_p95_ms=float(np.percentile(times,95)),
        optimization_max_ms=float(times.max()),latency_role="concurrent-run diagnostic; serial timing reported separately",
        trace_sha256=digest(trace_path),solver_log_sha256=digest(solver_path))
    for key in ("controller_logic_ms","local_pipeline_ms"):
        measured=np.asarray(trace[key])[5:]
        result[key.replace("_ms","_mean_ms")]=float(measured.mean())
        result[key.replace("_ms","_p95_ms")]=float(np.percentile(measured,95))
    return result


def formal_worker(args,options):
    from iotexp.supplement_common import verify_v2,load_base_backend,make_scenario_backend,scenarios
    verify_v2()
    protocol_path=args.out/"solver_protocol.json"
    protocol=json.loads(protocol_path.read_text())
    if protocol["options"]!=asdict(options): raise ValueError("Solver configuration differs from frozen protocol")
    for relative,expected in protocol["source_files"].items():
        if digest(ROOT/relative)!=expected: raise ValueError("Frozen solver source changed")
    prefix="oracle" if args.oracle else "mpc"
    target=args.out/f"{prefix}_worker{args.worker_id}.csv"
    if target.exists(): raise FileExistsError(target)
    partial=target.with_suffix(".in_progress.csv")
    with partial.open("x",newline="",encoding="utf-8") as stream:
        writer=None; count=0; index=0
        for name,seed,surge in scenarios():
            if args.oracle and (name=="legacy" or surge): continue
            backend=load_base_backend() if name=="legacy" else make_scenario_backend(seed,surge)
            dataset_hashes=backend.dataset_hashes()
            for day in range(27):
                selected=index%args.workers==args.worker_id; index+=1
                if not selected: continue
                relative=Path("traces")/prefix/name/f"day{day:02d}.npz"
                solver_relative=Path("solver_logs")/prefix/name/f"day{day:02d}.jsonl"
                result=evaluate_control_episode(backend,"test",day,options,args.out/relative,
                                                args.out/solver_relative,args.oracle)
                row=dict(method="perfect_information_flow_mpc" if args.oracle else "forecast_flow_mpc",
                    scenario=name,dataset_seed=seed,surge=surge,environment_seed=day,
                    physical_day=153+day,corruption_replicate=11,condition="nominal",
                    oracle_future_requests=bool(args.oracle),forecast_model="v2_edge_stgru_seed0",
                    dataset_hashes=json.dumps(dataset_hashes,sort_keys=True),
                    trace_path=relative.as_posix(),solver_path=solver_relative.as_posix(),**result)
                if writer is None: writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);stream.flush();count+=1
                print(f"{prefix} worker{args.worker_id}: {name} day{day} complete ({count})",flush=True)
        stream.flush();os.fsync(stream.fileno())
    os.replace(partial,target)


def launch_formal(args,options):
    from iotexp.supplement_common import verify_v2
    manifest=verify_v2()
    parent=ROOT/"runs/supplement_v3/protocol.json"
    if not parent.exists(): raise RuntimeError("Shared supplementary protocol has not been frozen")
    protocol_path=args.out/"solver_protocol.json"
    source_files={p:digest(ROOT/p) for p in ["iotexp/control_supplement.py","scripts/run_control_supplement.py",
                                             "tests/test_control_supplement.py","iotexp/supplement_common.py"]}
    if not protocol_path.exists():
        write_json(protocol_path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),options=asdict(options),
            shared_protocol_sha256=digest(parent),v2_source_sha256=manifest["source_sha256"],source_files=source_files,
            controller="finite-horizon mixed-integer time-expanded flow MPC; receding first action",
            objective="sum predicted queue mean /25 + .02*mean positive overload + .025*resource_km, original reward scale",
            forecasts="frozen edge_stgru seed0, linear interpolation at lead3/5; no test-truth inputs",
            commitments="own acknowledged dispatch history reconciled with exact trusted resource ledger",
            execution="optimized origin-destination flows; same original physical service and transport semantics; PPO uses fixed nearest-source pairing",
            horizons=6,mpc_episodes=189,oracle_episodes=81,
            oracle_scope="new nominal domains only; exact future service-request path, same bounded H6 algorithm, NOT an optimal lower bound",
            infeasible_fallback="hold all existing resources and immutable commitments; never recall in-transit units",
            solver_status="retain status, incumbent, bound, residual, gap, nodes and elapsed time every step; time limit is solver soft limit",
            timing="formal concurrent times diagnostic only; serial uncached validation timing after all workers finish"))
    else:
        old=json.loads(protocol_path.read_text())
        if old["options"]!=asdict(options) or old["source_files"]!=source_files:
            raise ValueError("Cannot alter frozen solver protocol")
    prefix="oracle" if args.oracle else "mpc"
    workers=[]
    for wid in range(args.workers):
        log=(args.out/f"{prefix}_worker{wid}.log").open("x",encoding="utf-8")
        command=[sys.executable,str(Path(__file__).resolve()),"--stage","worker","--out",str(args.out),
                 "--time-limit",str(options.time_limit_seconds),"--gap",str(options.relative_gap),
                 "--workers",str(args.workers),"--worker-id",str(wid)]
        if args.oracle:command.append("--oracle")
        workers.append((subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
                                        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)),log))
    failed=[]
    for child,log in workers:
        status=child.wait();log.close()
        if status:failed.append(status)
    if failed:raise RuntimeError(f"MPC worker failed: {failed}; retained per-episode outputs and logs")
    rows=[]
    for wid in range(args.workers):
        with (args.out/f"{prefix}_worker{wid}.csv").open(newline="",encoding="utf-8") as stream:rows.extend(csv.DictReader(stream))
    expected=81 if args.oracle else 189
    if len(rows)!=expected or len({(r["scenario"],r["environment_seed"]) for r in rows})!=expected:
        raise ValueError("Incomplete or duplicated MPC evaluation matrix")
    rows.sort(key=lambda r:(r["scenario"],int(r["environment_seed"])))
    target=args.out/f"{prefix}_episodes.csv"
    with target.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f"Completed {prefix}: {len(rows)} episodes; {target}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=Path("runs/scenic_rebuild_v2"))
    parser.add_argument("--out",type=Path,default=Path("runs/supplement_v3/control"))
    parser.add_argument("--stage",choices=["validation","formal","worker","timing"],default="validation")
    parser.add_argument("--time-limit",type=float,default=.25)
    parser.add_argument("--gap",type=float,default=.01)
    parser.add_argument("--oracle",action="store_true")
    parser.add_argument("--workers",type=int,choices=[1,2],default=2)
    parser.add_argument("--worker-id",type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    options=MPCConfig(time_limit_seconds=args.time_limit,relative_gap=args.gap)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=="formal":return launch_formal(args,options)
    if args.stage=="worker":return formal_worker(args,options)
    backend=load_original_backend(args.base)
    name=f"{args.stage}_t{args.time_limit:g}_gap{args.gap:g}"+("_oracle" if args.oracle else "")
    result=evaluate_control_episode(backend,"validation",0,options,
                                    args.out/name/"trace.npz",args.out/name/"solver.jsonl",args.oracle)
    write_json(args.out/name/"result.json",dict(options=asdict(options),split="validation",day=0,**result))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
