"""V4 development only, then one selection confirmation; final tests are separate."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from iotexp.forecast_improvement import Anchor,GraphResidual,ridge_features,feature_tensors,neural_prediction
from iotexp.forecast_supplement import setup_runtime,make_model,predict
from iotexp.scenic_forecast import fit_training_stats,build_windows
from iotexp.util import digest,write_json

OUT=ROOT/'runs/improvement_v4/forecast'
CANDIDATES=[{'id':'S0','kind':'anchor'},*[{'id':f'R{a}','kind':'ridge','alpha':a} for a in [10,100,1000]],
 {'id':'G_mse','kind':'mlp','hidden':64,'loss':'mse'},
 {'id':'G_huber','kind':'mlp','hidden':64,'loss':'raw_huber'},
 {'id':'T_huber','kind':'gru','hidden':64,'loss':'raw_huber'}]
CONF=dict(device='cuda',torch_threads=1,epochs=100,patience=12,batch_size=128,period=72,history=12,horizons=[1,2,4,6])

def windows(data,days):
    pos=np.arange(14,81);hist=pos[:,None]-np.arange(11,-1,-1)[None]
    return dict(raw=data['inflow'][days][:,hist].reshape(-1,12,24).astype(np.float32),
                y=data['inflow'][days][:,pos[:,None]+np.asarray([1,2,4,6])[None]].transpose(0,1,3,2).reshape(-1,24,4).astype(np.float32),
                origin=np.tile(np.arange(67),len(days)),day=np.repeat(days,67))

def train_neural(spec,seed,anchor,training,development):
    path=OUT/'training'/spec['id']/f'seed{seed}';path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return json.loads((path/'complete.json').read_text(encoding='utf-8'))
    setup_runtime(CONF,seed);model=GraphResidual(anchor,spec['kind'],spec['hidden']).to(CONF['device'])
    train_f=anchor.features(training['raw'],training['origin']);dev_f=anchor.features(development['raw'],development['origin'])
    tensors=feature_tensors(train_f,CONF['device'])
    target=torch.as_tensor((training['y']-train_f['base'])/anchor.scale[None,:,None],device=CONF['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_epoch=0;logs=[];rng=np.random.default_rng(seed);began=time.perf_counter()
    for epoch in range(1,101):
        model.train();order=rng.permutation(len(target));total=0
        for start in range(0,len(order),128):
            indices=torch.as_tensor(order[start:start+128],device=CONF['device'])
            pred=model(**{k:v[indices] for k,v in tensors.items()})
            loss=F.mse_loss(pred,target[indices]) if spec['loss']=='mse' else F.smooth_l1_loss(pred*model.scale[None,:,None],target[indices]*model.scale[None,:,None],beta=1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())*len(indices)
        prediction=neural_prediction(model,dev_f,anchor,CONF['device']);score=float(np.abs(prediction-development['y']).mean())
        logs.append(dict(epoch=epoch,loss=total/len(order),development_mae=score))
        if score<best-1e-7:
            best=score;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0:print(spec['id'],seed,epoch,round(best,5),flush=True)
        if epoch-best_epoch>=12:break
    model.load_state_dict(state);prediction=neural_prediction(model,dev_f,anchor,CONF['device'])
    torch.save(dict(state_dict=state,spec=spec,seed=seed),path/'checkpoint.pt')
    np.savez_compressed(path/'development_predictions.npz',prediction=prediction)
    write_json(path/'log.json',logs)
    result=dict(**spec,seed=seed,best_epoch=best_epoch,epochs=epoch,development_mae=best,seconds=time.perf_counter()-began,
                checkpoint=str((path/'checkpoint.pt').relative_to(OUT)).replace('\\','/'),checkpoint_sha256=digest(path/'checkpoint.pt'))
    write_json(path/'complete.json',result);print('COMPLETE',result,flush=True);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise FileExistsError('Selection already frozen')
    protocol=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),candidates=CANDIDATES,
                  train_days=list(range(126)),development_days=list(range(126,144)),confirmation_days=list(range(144,153)),
                  selection='Lowest seed0 development mean MAE across four leads and all24 nodes; ties preserve candidate order. Fit selected neural family seeds1/2. Deploy mean of3 predictions; deterministic families one fit. Confirmation never reselects.',
                  runtime=CONF,information='12 observed inflow frames, clock and fixed graph; all fitted transforms training only',
                  previous_exposure='V3 baselines used all27 validation days. Confirmation split is a new-search discipline, not an historically untouched dataset.',
                  source_sha256={p:digest(ROOT/p) for p in ['iotexp/forecast_improvement.py','scripts/run_forecast_improvement.py']})
    if not (OUT/'development_protocol.json').exists():write_json(OUT/'development_protocol.json',protocol)
    else:
        old=json.loads((OUT/'development_protocol.json').read_text(encoding='utf-8'))
        assert old['source_sha256']==protocol['source_sha256']
    data_path=ROOT/'runs/scenic_rebuild_v2/assets/synthetic_dataset.npz'
    with np.load(data_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    anchor=Anchor.fit(data);anchor.save(OUT/'anchor.npz')
    training=windows(data,np.arange(126));development=windows(data,np.arange(126,144))
    for name,win in [('development',development)]:np.savez_compressed(OUT/(name+'_targets.npz'),truth=win['y'],day=win['day'],origin=win['origin'])
    ft=anchor.features(training['raw'],training['origin']);fd=anchor.features(development['raw'],development['origin'])
    x=ridge_features(ft);xv=ridge_features(fd);y=((training['y']-ft['base'])/anchor.scale[None,:,None]).reshape(len(x),-1)
    gram=x.T@x;xy=x.T@y;results=[]
    for spec in CANDIDATES:
        if spec['kind']=='anchor':prediction=fd['base'];result=dict(**spec,development_mae=float(np.abs(prediction-development['y']).mean()))
        elif spec['kind']=='ridge':
            regularizer=np.eye(x.shape[1])*spec['alpha'];regularizer[-1,-1]=0
            coef=np.linalg.solve(gram+regularizer,xy);prediction=np.maximum(fd['base']+(xv@coef).reshape(fd['base'].shape)*anchor.scale[None,:,None],0)
            file=OUT/(spec['id']+'.npz');np.savez_compressed(file,coefficients=coef)
            result=dict(**spec,development_mae=float(np.abs(prediction-development['y']).mean()),checkpoint=file.name,checkpoint_sha256=digest(file))
        else:
            result=train_neural(spec,0,anchor,training,development)
            prediction=np.load(OUT/Path(result['checkpoint']).parent/'development_predictions.npz')['prediction']
        results.append(result);np.savez_compressed(OUT/(spec['id']+'_development.npz'),prediction=prediction)
        write_json(OUT/'candidate_results.json',results);print('CANDIDATE',result,flush=True)
    selected=min(results,key=lambda r:r['development_mae']);checkpoints=[]
    if selected['kind'] in ('mlp','gru'):
        spec=next(s for s in CANDIDATES if s['id']==selected['id'])
        for seed in [0,1,2]:
            r=train_neural(spec,seed,anchor,training,development)
            checkpoints.append(dict(seed=seed,path=r['checkpoint'],sha256=r['checkpoint_sha256']))
    elif selected.get('checkpoint'):checkpoints=[dict(seed=None,path=selected['checkpoint'],sha256=selected['checkpoint_sha256'])]
    selection=dict(selected=selected,deployed_checkpoints=checkpoints,selected_before_confirmation=True,
                   created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),anchor_sha256=digest(OUT/'anchor.npz'),
                   protocol_sha256=digest(OUT/'development_protocol.json'))
    write_json(OUT/'selection.json',selection)
    from iotexp.forecast_improvement import ImprovedForecaster
    candidate=ImprovedForecaster(OUT,'cuda' if checkpoints and selected['kind'] in ('mlp','gru') else 'cpu')
    confirmation=windows(data,np.arange(144,153));pred=candidate.predict(confirmation['raw'],confirmation['origin'])
    np.savez_compressed(OUT/'confirmation.npz',prediction=pred,truth=confirmation['y'],day=confirmation['day'],origin=confirmation['origin'])
    report=dict(selected=selected['id'],development_seed0_mae=selected['development_mae'],confirmation_mae=float(np.abs(pred-confirmation['y']).mean()),
                confirmation_horizon_mae=np.abs(pred-confirmation['y']).mean((0,1)).tolist(),final_data_generated=False)
    write_json(OUT/'confirmation_report.json',report);print('SELECTION',report,flush=True)

if __name__=='__main__':main()
