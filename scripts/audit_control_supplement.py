"""Independent vector/FIFO replay of saved supplementary control flows.

Does not call the controller or ScenicEnvironment.step; changes no run outputs.
Writes audit artifacts only after a complete successful replay.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from iotexp.supplement_common import scenarios,load_base_backend,make_scenario_backend,verify_v2
from iotexp.util import digest,write_json


def close(x,y,name):
    if not np.allclose(x,y,rtol=1e-7,atol=1e-5,equal_nan=True):raise AssertionError(name)


def audit(out,oracle=False,partial=False):
    started=time.perf_counter();verify_v2()
    prefix="oracle" if oracle else "mpc"
    inputs=list(out.glob(f"{prefix}_worker*.in_progress.csv")) if partial else [out/f"{prefix}_episodes.csv"]
    table=pd.concat([pd.read_csv(p) for p in inputs],ignore_index=True)
    applicable=[s for s in scenarios() if not oracle or (s[0]!="legacy" and not s[2])]
    expected={(name,day) for name,_,_ in applicable for day in range(27)}
    if partial:
        assert set(zip(table.scenario,table.environment_seed)).issubset(expected)
    else:
        assert len(table)==len(expected)
        assert set(zip(table.scenario,table.environment_seed))==expected
    assert not table.duplicated(["scenario","environment_seed"]).any()
    solver=[];total_waits=0;arrivals_total=0
    for name,seed,surge in applicable:
        if not (table.scenario==name).any():continue
        backend=load_base_backend() if name=="legacy" else make_scenario_backend(seed,surge)
        cfg,data=backend.cfg,backend.data
        totals=np.array(cfg["resource_totals"]);holding=np.zeros((4,24),int)
        holding[:,:16]=np.asarray(cfg["holding_per_service_node"])[:,None]
        travel=np.ceil((data["shortest_distance_m"][None]/np.asarray(cfg["resource_speed_m_per_min"])[:,None,None]+
                        np.asarray(cfg["resource_setup_minutes"])[:,None,None])*
                       cfg["travel_time_multiplier"]/cfg["decision_minutes"]).astype(int)
        np.einsum("kii->ki",travel)[:]=0
        for row in table[table.scenario==name].itertuples():
            assert json.loads(row.dataset_hashes)==backend.dataset_hashes()
            path,logpath=out/row.trace_path,out/row.solver_path
            assert digest(path)==row.trace_sha256 and digest(logpath)==row.solver_log_sha256
            steps=[json.loads(line) for line in logpath.read_text().splitlines()]
            assert len(steps)==72 and all(bool(r["oracle_future_requests"])==oracle for r in steps)
            inventory=np.zeros((4,24),int)
            for k,total in enumerate(totals):
                inventory[k,:16]=total//16;inventory[k,:total%16]+=1
            reserved=np.zeros_like(inventory);queue=np.zeros_like(inventory)
            batches=np.zeros((4,24,73),int);trips=[];waits=[];area=0.;movement=0.;served_count=0;arrived=0
            with np.load(path) as z:
                assert np.array_equal(z["decision_time"],np.arange(72))
                assert (np.isnan(z["source_receipt"])|(z["source_receipt"]<=np.arange(72)[:,None,None])).all()
                grid=np.arange(72)[:,None,None]-np.arange(11,-1,-1)[None,:,None]
                assert (np.isnan(z["source_event"])|(z["source_event"]<=grid)).all()
                for t in range(72):
                    eta=np.zeros_like(inventory)
                    for k,_,j,n,l in trips:eta[k,j]+=n*l
                    close(z["ledger_before"][t,:288],np.r_[inventory.ravel(),reserved.ravel(),eta.ravel()],"before ledger")
                    close(z["observed"][t,-1,:,1:].T,queue,"nominal causal queue")
                    close(z["observed"][t,-1,:,0],data["inflow"][153+row.environment_seed,14+t],"nominal causal inflow")
                    x=z["transfers"][t]
                    assert (x>=0).all() and np.array_equal(x,np.rint(x))
                    assert (np.diagonal(x,axis1=1,axis2=2)==0).all()
                    assert (x.sum(2)<=inventory).all()
                    inventory-=x.sum(2);reserved+=x.sum(1)
                    assert ((inventory+reserved)<=holding).all()
                    for k,i,j in zip(*np.nonzero(x)):trips.append([int(k),int(i),int(j),int(x[k,i,j]),int(travel[k,i,j])])
                    cost=float((x*data["shortest_distance_m"][None]).sum()/1000)
                    movement+=cost;close(cost,z["movement_cost"][t],"step movement")
                    step_area=queue.sum()*10;area+=step_area;close(step_area,z["queue_area_min"][t],"step area")
                    requests=data["demand"][153+row.environment_seed,15+t];arrived+=requests.sum()
                    queue+=requests;batches[:,:,t+1]+=requests
                    capacity=inventory*np.asarray(cfg["service_per_resource"])[:,None]
                    capacity[:,:16]+=np.asarray(cfg["base_service"])[:,None]
                    served=np.minimum(capacity,queue);queue-=served;served_count+=served.sum()
                    for k,n in zip(*np.nonzero(served)):
                        remaining=int(served[k,n])
                        for age in range(t+2):
                            amount=min(remaining,int(batches[k,n,age]));batches[k,n,age]-=amount
                            waits.extend([(t+1-age)*10]*amount);remaining-=amount
                            if not remaining:break
                    next_trips=[]
                    for k,i,j,n,l in trips:
                        if l==1:inventory[k,j]+=n;reserved[k,j]-=n
                        else:next_trips.append([k,i,j,n,l-1])
                    trips=next_trips
                    assert np.array_equal((inventory+reserved).sum(1),totals)
                    close(queue.sum(0),z["occupancy"][t],"post queue")
                    close(queue.sum(),z["unserved_count"][t],"post backlog")
                    reward=-(queue[:,:16].sum()/16/cfg["queue_reward_scale"]+cfg["movement_penalty"]*cost+
                              cfg["overload_penalty"]*np.maximum(queue[:,:16].sum(0)-cfg["queue_capacity_per_service_node"],0).mean())
                    close(reward,z["reward"][t],"step reward")
                    h=steps[t]["horizon"]
                    if not oracle:
                        projected=np.stack([np.interp(np.arange(1,h+1),[1,2,4,6],node) for node in z["forecast"][t]],1)
                        expected_demand=projected[:,None,:]*np.asarray(cfg["request_probability"])[None,:,None]
                        expected_demand[:,:,16:]=0
                    else:expected_demand=data["demand"][153+row.environment_seed,15+t:15+t+h]
                    close(z["forecast_requests"][t,:h],expected_demand,"forecast/oracle isolation")
                close(z["served_waits_min"],waits,"raw FIFO wait sample replay")
                close(z["terminal_queue_batches"],batches,"terminal batches")
                close(area,row.waiting_area_min,"episode area")
                close(area/arrived,row.restricted_mean_wait_min,"restricted mean")
                close(sum(waits),row.wait_sum_min,"served wait sum")
                close(movement,row.movement_cost,"movement total")
                assert served_count==row.served_count and arrived==row.arrivals_count
                assert queue.sum()==row.unserved_count and served_count+queue.sum()==arrived
            solver.extend(steps);total_waits+=len(waits);arrivals_total+=int(arrived)
        print(f"Replayed {prefix} {name}",flush=True)
    gaps=[r["mip_gap"] for r in solver if r["mip_gap"] is not None]
    report=dict(status="PARTIAL_PASS" if partial else "PASS",method=prefix,episodes=len(table),expected_episodes=len(expected),steps=len(solver),
        raw_fifo_request_waits_replayed=total_waits,request_exposures=arrivals_total,
        fallback_steps=sum(r["fallback"] for r in solver),
        solver_status_counts={str(i):sum(r["status"]==i for r in solver) for i in range(5)},
        mip_gap_quantiles=dict(zip(["min","median","p90","p95","max"],map(float,np.percentile(gaps,[0,50,90,95,100])))),
        checks=["observed unique subset of expected scenario/day matrix" if partial else "complete paired scenario/day matrix",
                "dataset and episode hashes","causal event/receipt timestamps",
                "nominal observed queues and inflows independently replayed","resource inventory, reservations and ETA ledger",
                "integer executable dispatch and holding constraints","exact service-arrival ordering and FIFO waits",
                "terminal censored queue accounting","all rewards and movement cost","ordinary forecast/oracle input isolation"],
        elapsed_seconds=time.perf_counter()-started,episodes_sha256=None if partial else digest(out/f"{prefix}_episodes.csv"))
    suffix="_partial" if partial else ""
    write_json(out/f"{prefix}{suffix}_independent_audit.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"runs/supplement_v3/control");p.add_argument("--oracle",action="store_true");p.add_argument("--partial",action="store_true")
    a=p.parse_args();audit(a.out,a.oracle,a.partial)
