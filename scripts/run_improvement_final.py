"""V4 sealed evaluation: no fitting or selection entry point in final modes."""
import os,sys,json,time,copy,argparse,csv,itertools
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from iotexp.improvement_runtime import ImprovedBackend,ForecastRuntime,V4
from iotexp.forecast_improvement import Anchor,ImprovedForecaster
from iotexp.forecast_supplement import setup_runtime
from iotexp.scenic import build_dataset,scenic_config
from iotexp.supplement_common import base_config,verify_v2,V3,SupplementBackend
from iotexp.control_supplement import FlowMPC,MPCConfig
from iotexp.transport_rollout import TransportRollout
from iotexp.improvement_controllers import SeasonalMPC,StaticController
from iotexp.engine import evaluate_episode,make_policy
from iotexp.telemetry import Condition
from iotexp.scenic_forecast import build_windows,_ridge_features,_unscale
from iotexp.util import write_json,digest
from run_forecast_improvement import windows
import run_control_supplement as runner

def protocol():return json.loads((V4/'experiment_plan.json').read_text())
def configuration(seed,surge):
    cfg=copy.deepcopy(base_config());cfg['scenic']['dataset_seed']=seed
    if surge:
        for key in ['gate_base','gate_peak']:cfg['scenic'][key]=[1.6*x for x in cfg['scenic'][key]]
        cfg['scenic']['event_amplitude']*=1.6
    return cfg

def freeze():
    verify_v2()
    if (V4/'final_freeze.json').exists():raise FileExistsError('Final freeze exists')
    plan=protocol()
    sources=[f'iotexp/{n}.py' for n in ['forecast_improvement','forecast_supplement','improvement_runtime','transport_rollout','improvement_controllers','control_supplement','supplement_common','scenic','scenic_forecast','engine','telemetry','metrics']]
    sources+=['scripts/run_improvement_final.py','scripts/run_control_supplement.py','scripts/run_forecast_improvement.py','scripts/select_forecast_combination.py','tests/test_improvement_v4.py']
    assets=[V4/'experiment_plan.json',V4/'forecast/selection.json',V4/'forecast/combination_selection.json',V4/'control_development/selection.json',V4/'forecast/anchor.npz']
    assets+=list((V4/'forecast/training/T_huber').glob('seed*/checkpoint.pt'))
    assets+=list((V3/'forecast/training').glob('*/C*/seed*/checkpoint.pt'))
    assets+=[V3/'forecast'/n for n in ['selected_models.json','training_stats.npz','ridge.npz','historical_average.npz']]
    assets+=list((V3/'policies/edge_stgru/checkpoints').glob('F[04]_seed*/best.pt'))
    write_json(V4/'final_freeze.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p:digest(ROOT/p) for p in sources},asset_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in assets},
        full_control_methods=['static','gwn_mpc6','combined_mpc6','rollout','ppo_f0','ppo_f4'],
        matched_horizon_comparator=dict(method='combined_mpc12',test_offsets=[0,12,24],seeds=plan['final_seeds'],conditions=plan['conditions'],
            interpretation='Prespecified 21 episodes per condition; compare other methods on exactly same subset. One-second solver budget and all gaps retained.'),
        policy_replication='F0/F4 all10 existing independently trained seeds; summarize within world, do not count policy seeds as new worlds',
        predictor_rule='Each neural family uses three-seed prediction mean, identical ensemble convention. Forecast scores also retained per seed in prior development archives.',
        compute='MPC one thread, 1s soft solve limit,1% relative gap; rollout deterministic. Parallel episode times descriptive only; separate serial uncached timing.',
        no_more_selection=True))

def verify():
    f=json.loads((V4/'final_freeze.json').read_text())
    for p,h in {**f['source_sha256'],**f['asset_sha256']}.items():
        if digest(ROOT/p)!=h:raise RuntimeError('Frozen artifact changed: '+p)
    verify_v2();return f

def forecast():
    verify();setup_runtime(dict(torch_threads=1),0)
    out=V4/'final_forecast';out.mkdir(exist_ok=True)
    runtimes={};rows=[]
    for surge,seed in itertools.product([False,True],protocol()['final_seeds']):
        tag=f'{seed}_{"surge" if surge else "nominal"}';target=out/tag
        if (target/'complete.json').exists():continue
        target.mkdir(exist_ok=True);data=build_dataset(scenic_config(configuration(seed,surge)))
        dataset=V4/'final_datasets'/f'{tag}.npz';dataset.parent.mkdir(exist_ok=True)
        if not dataset.exists():np.savez_compressed(dataset,**{k:v for k,v in data.items() if k!='config'})
        win=windows(data,np.arange(153,180));truth=win['y'];preds={}
        for name in ['lstm','gru','edge_stgru','graph_wavenet','stid','itransformer','residual']:
            if name not in runtimes:runtimes[name]=ForecastRuntime(data,name,'cuda')
            preds[name]=runtimes[name].predict(win['raw'],win['origin'])
        w=json.loads((V4/'forecast/combination_selection.json').read_text())['residual_weight']
        preds['combined']=w*preds['residual']+(1-w)*preds['graph_wavenet']
        anchor=runtimes['residual'].residual.anchor
        preds['anchor']=anchor.features(win['raw'],win['origin'])['base']
        preds['persistence']=np.repeat(win['raw'][:,-1,:,None],4,axis=2)
        stats=runtimes['graph_wavenet'].stats;oldwin=build_windows(data,np.arange(153,180),dict(history=12,horizons=[1,2,4,6]),stats)
        with np.load(V3/'forecast/ridge.npz') as z:coef=z['coefficients']
        preds['ridge']=_unscale((_ridge_features(oldwin,72)@coef).reshape(truth.shape),stats)
        np.savez_compressed(target/'predictions.npz',truth=truth,day=win['day'],origin=win['origin'],**preds)
        scores=[]
        for name,pred in preds.items():
            error=pred-truth
            scores.append(dict(model=name,dataset_seed=seed,surge=surge,mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
                wape=float(100*np.abs(error).sum()/truth.sum()),horizon_mae=np.abs(error).mean((0,1)).tolist()))
        write_json(target/'complete.json',dict(scores=scores,dataset_sha256=digest(dataset),prediction_sha256=digest(target/'predictions.npz')))
        print('FORECAST',tag,{s['model']:round(s['mae'],4) for s in scores},flush=True)

def worker(wid,workers):
    verify();torch.set_num_threads(1)
    out=V4/'final_control';out.mkdir(exist_ok=True)
    anchor=Anchor.load(V4/'forecast/anchor.npz');sel=json.loads((V4/'control_development/selection.json').read_text())['selected']
    index=0
    for surge,seed in itertools.product([False,True],protocol()['final_seeds']):
        tag=f'{seed}_{"surge" if surge else "nominal"}';cfg=configuration(seed,surge)
        base=SupplementBackend(cfg);combined=ImprovedBackend(cfg,'combined');gwn=ImprovedBackend(cfg,'graph_wavenet')
        for b in [combined,gwn]:b.prime_causal_cache(153+np.array(protocol()['control_test_offsets']))
        # Frozen PPO architecture and weights; forecast remains their training predictor.
        policies={}
        for method in ['F0','F4']:
            for training_seed in range(10):
                m=make_policy(base,cfg,method,training_seed)
                state=torch.load(V3/'policies/edge_stgru/checkpoints'/f'{method}_seed{training_seed}'/'best.pt',map_location='cpu',weights_only=True)
                m.load_state_dict(state['state_dict']);policies[(method,training_seed)]=m
        for day in protocol()['control_test_offsets']:
            methods=['static','rollout','gwn_mpc6','combined_mpc6']+(['combined_mpc12'] if day in [0,12,24] else [])+['ppo_f0','ppo_f4']
            for method in methods:
                mine=index%workers==wid;index+=1
                if not mine:continue
                folder=out/tag/method/f'day{day:02d}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').exists():continue
                results=[]
                if method.startswith('ppo'):
                    family='F0' if method=='ppo_f0' else 'F4'
                    for training_seed in range(10):
                        dest=folder/f'seed{training_seed}.json'
                        if dest.exists():r=json.loads(dest.read_text())
                        else:
                            r=evaluate_episode(policies[(family,training_seed)],base,cfg,'test',day,Condition(),11,folder/f'seed{training_seed}.npz')
                            r['training_seed']=training_seed;write_json(dest,r)
                        results.append(r)
                else:
                    b=gwn if method=='gwn_mpc6' else combined
                    h=sel['horizon'] if method=='rollout' else (12 if method=='combined_mpc12' else 6)
                    if method=='rollout':runner.FlowMPC=lambda c,d,o:TransportRollout(c,d,o,anchor,sel['terminal_weight'])
                    elif method=='static':runner.FlowMPC=StaticController
                    elif method=='combined_mpc12':runner.FlowMPC=lambda c,d,o:SeasonalMPC(c,d,o,anchor,0)
                    else:runner.FlowMPC=FlowMPC
                    r=runner.evaluate_control_episode(b,'test',day,MPCConfig(horizon=h,time_limit_seconds=1),folder/'trace.npz',folder/'solver.jsonl')
                    results=[r]
                write_json(folder/'complete.json',dict(method=method,dataset_seed=seed,surge=surge,day=day,physical_day=153+day,results=results,dataset_hashes=base.dataset_hashes()))
                print(wid,tag,method,day,round(np.mean([r['restricted_mean_wait_min'] for r in results]),3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','forecast','worker']);p.add_argument('--worker-id',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='forecast':forecast()
    else:worker(a.worker_id,a.workers)
