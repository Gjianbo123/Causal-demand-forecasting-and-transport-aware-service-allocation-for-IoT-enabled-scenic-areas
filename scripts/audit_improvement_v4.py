"""Independent replay and numerical audit; never edits evaluated artifacts."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from iotexp.util import write_json,digest
from iotexp.forecast_improvement import Anchor
from run_improvement_final import verify,configuration,V4

def close(a,b,tag,atol=1e-5):
    if not np.allclose(a,b,atol=atol,rtol=1e-7,equal_nan=True):raise AssertionError(tag)

def replay(record,path,data,cfg,anchor):
    day=record['physical_day'];row=record['results'][0]
    travel=np.ceil((data['shortest_distance_m'][None]/np.array(cfg['resource_speed_m_per_min'])[:,None,None]+np.array(cfg['resource_setup_minutes'])[:,None,None])/10).astype(int)
    np.einsum('kii->ki',travel)[:]=0
    holding=np.zeros((4,24),int);holding[:,:16]=np.array(cfg['holding_per_service_node'])[:,None]
    inv=np.zeros_like(holding)
    for k,total in enumerate(cfg['resource_totals']):inv[k,:16]=total//16;inv[k,:total%16]+=1
    reserved=np.zeros_like(inv);q=np.zeros_like(inv);batches=np.zeros((4,24,73),int)
    trips=[];waits=[];area=0.;movement=0.;arrived=0;overload=0
    assert digest(path/'trace.npz')==row['trace_sha256'];assert digest(path/'solver.jsonl')==row['solver_log_sha256']
    logs=[json.loads(line) for line in (path/'solver.jsonl').read_text().splitlines()]
    with np.load(path/'trace.npz') as z:
        assert len(logs)==72 and np.array_equal(z['decision_time'],np.arange(72))
        assert (np.isnan(z['source_receipt'])|(z['source_receipt']<=np.arange(72)[:,None,None])).all()
        grid=np.arange(72)[:,None,None]-np.arange(11,-1,-1)[None,:,None]
        assert (np.isnan(z['source_event'])|(z['source_event']<=grid)).all()
        for t in range(72):
            eta=np.zeros_like(inv)
            for k,i,j,n,l in trips:eta[k,j]+=n*l
            close(z['ledger_before'][t,:288],np.r_[inv.ravel(),reserved.ravel(),eta.ravel()],'ledger')
            close(z['observed'][t,-1,:,1:].T,q,'causal queues')
            close(z['observed'][t,:, :,0],data['inflow'][day,t+3:t+15],'causal history')
            x=z['transfers'][t];assert (x>=0).all() and np.array_equal(x,np.rint(x))
            assert not np.diagonal(x,axis1=1,axis2=2).any();assert (x.sum(2)<=inv).all()
            inv-=x.sum(2);reserved+=x.sum(1);assert ((inv+reserved)<=holding).all()
            for k,i,j in zip(*np.nonzero(x)):trips.append([k,i,j,int(x[k,i,j]),int(travel[k,i,j])])
            cost=float((x*data['shortest_distance_m'][None]).sum()/1000);movement+=cost
            close(cost,z['movement_cost'][t],'movement');area+=q.sum()*10
            close(q.sum()*10,z['queue_area_min'][t],'area')
            demand=data['demand'][day,t+15];arrived+=demand.sum();q+=demand;batches[:,:,t+1]+=demand
            cap=inv*np.array(cfg['service_per_resource'])[:,None];cap[:,:16]+=np.array(cfg['base_service'])[:,None]
            served=np.minimum(cap,q);q-=served
            for k,n in zip(*np.nonzero(served)):
                left=int(served[k,n])
                for birth in range(t+2):
                    take=min(left,int(batches[k,n,birth]));batches[k,n,birth]-=take
                    waits.extend([(t+1-birth)*10]*take);left-=take
                    if not left:break
            next_trips=[]
            for k,i,j,n,l in trips:
                if l==1:inv[k,j]+=n;reserved[k,j]-=n
                else:next_trips.append([k,i,j,n,l-1])
            trips=next_trips;assert np.array_equal((inv+reserved).sum(1),cfg['resource_totals'])
            close(q.sum(0),z['occupancy'][t],'occupancy');overload+=(q.sum(0)[:16]>50).sum()
            close(q.sum(),z['unserved_count'][t],'backlog')
            reward=-(q.sum()/16/25+.025*cost+.02*np.maximum(q.sum(0)[:16]-50,0).mean())
            close(reward,z['reward'][t],'reward')
            h=logs[t]['horizon'];forecast=z['forecast'][t]
            inflow=np.stack([np.interp(np.arange(1,h+1),[1,2,4,6],f) for f in forecast],1)
            if h>6:
                ref=anchor.template[t+6-anchor.first];factor=np.clip(forecast[:,-1]/np.maximum(ref,.25),.05,5)
                inflow[6:]=anchor.template[t+np.arange(7,h+1)-anchor.first]*factor[None]
            expected=inflow[:,None]*np.array(cfg['request_probability'])[None,:,None];expected[:,:,16:]=0
            close(expected,z['forecast_requests'][t,:h],'causal demand interpolation and tail')
        close(waits,z['served_waits_min'],'FIFO waits');close(batches,z['terminal_queue_batches'],'terminal cohorts')
        close(sum(waits),row['wait_sum_min'],'wait sum');close(area,row['waiting_area_min'],'restricted area')
        close(area/arrived,row['restricted_mean_wait_min'],'primary endpoint');close(movement,row['movement_cost'],'movement total')
        assert len(waits)==row['served_count'] and arrived==row['arrivals_count'] and q.sum()==row['unserved_count']
        assert len(waits)+q.sum()==arrived and overload==row['visitor_overload_count']
        close(area,sum(waits)+(batches*(72-np.arange(73))).sum()*10,'censored waiting identity')
    return len(waits)

def main():
    verify();begin=time.perf_counter();anchor=Anchor.load(V4/'forecast/anchor.npz')
    partial='--partial' in sys.argv
    datasets={};forecast_count=0;controls=0;policies=0;waits=0
    for path in sorted((V4/'final_forecast').glob('*/complete.json')):
        record=json.loads(path.read_text());dataset=V4/'final_datasets'/f'{path.parent.name}.npz'
        assert digest(dataset)==record['dataset_sha256'] and digest(path.parent/'predictions.npz')==record['prediction_sha256']
        with np.load(dataset) as z:data={k:z[k] for k in z.files}
        datasets[path.parent.name]=data
        with np.load(path.parent/'predictions.npz') as z:
            truth=np.stack([data['inflow'][d,t+14+np.array([1,2,4,6])].T for d,t in zip(z['day'],z['origin'])])
            close(z['truth'],truth,'targets');assert len(truth)==27*67
            for row in record['scores']:
                error=z[row['model']]-truth;close(np.abs(error).mean(),row['mae'],'forecast mae')
                close(np.sqrt((error**2).mean()),row['rmse'],'forecast rmse');close(np.abs(error).mean((0,1)),row['horizon_mae'],'forecast leads',atol=5e-5)
                close(100*np.abs(error).sum()/truth.sum(),row['wape'],'forecast WAPE')
                assert np.isfinite(error).all();forecast_count+=1
    for path in sorted((V4/'final_control').glob('*/*/day*/complete.json')):
        record=json.loads(path.read_text());tag=path.parents[2].name;data=datasets[tag]
        cfg=configuration(record['dataset_seed'],record['surge'])['scenic']
        if record['method'].startswith('ppo'):
            assert len(record['results'])==10
            for r in record['results']:
                with np.load(path.parent/f"seed{r['training_seed']}.npz") as z:
                    assert len(z['decision_time'])==72 and not z['executable_infeasible'].any()
                    close(z['served_waits_min'].sum(),r['wait_sum_min'],'PPO raw waits')
                    assert len(z['served_waits_min'])==r['served_count']
                    assert r['served_count']+r['unserved_count']==r['arrivals_count']
                    assert r['arrivals_count']==data['demand'][record['physical_day'],15:87].sum()
                    close(r['waiting_area_min']/r['arrivals_count'],r['restricted_mean_wait_min'],'PPO primary')
                    policies+=1
        else:waits+=replay(record,path.parent,data,cfg,anchor);controls+=1
    if not partial:assert forecast_count==14*11 and controls==546 and policies==2520,(forecast_count,controls,policies)
    write_json(V4/('PARTIAL_AUDIT.json' if partial else 'INDEPENDENT_AUDIT.json'),dict(status='PARTIAL_PASS' if partial else 'PASS',forecast_scenario_model_pairs=forecast_count,full_flow_episodes_replayed=controls,
        policy_episodes_aggregate_checked=policies,fifo_wait_samples_replayed=waits,seconds=time.perf_counter()-begin,
        limits='PPO compact traces support aggregate checks, not independent action replay; original unchanged PPO engine has separate v3 full-trace audits. Forecast errors recomputed in float64; 5e-5 absolute tolerance for float32 horizon reductions (observed maximum discrepancy3.48e-5), 1e-5 otherwise. No claim of rerunning training here.',
        checks=['frozen source and checkpoint hashes','forecast target indices and all reported errors','causal observation timestamps','complete integer resource and transport replay','FIFO waits and terminal censoring','resource conservation and holding','same realized demand across controllers']))
    print('AUDIT', 'PARTIAL_PASS' if partial else 'PASS',forecast_count,controls,policies,flush=True)

if __name__=='__main__':main()
