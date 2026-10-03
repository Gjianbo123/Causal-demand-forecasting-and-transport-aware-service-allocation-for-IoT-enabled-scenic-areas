"""Development-only fixed-grid combination; frozen component models."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from iotexp.util import write_json,digest
V3=ROOT/'runs/supplement_v3/forecast';OUT=ROOT/'runs/improvement_v4/forecast'

def main():
    if (OUT/'combination_selection.json').exists():raise FileExistsError('Combination frozen')
    specs=json.loads((V3/'selected_models.json').read_text())
    truth=np.load(OUT/'development_targets.npz')['truth'];n=len(truth)
    rows=[];preds={}
    for name in ['lstm','gru','edge_stgru','graph_wavenet','stid','itransformer']:
        ps=[np.load(V3/Path(s['checkpoint']).parent/'validation_predictions.npz')['prediction'] for s in specs if s['model']==name]
        preds[name]=np.mean(ps,axis=0)
        rows.append(dict(model=name+'_ensemble3',development_mae=float(np.abs(preds[name][:n]-truth).mean())))
    selected=json.loads((OUT/'selection.json').read_text())
    residual=np.mean([np.load(OUT/Path(s['path']).parent/'development_predictions.npz')['prediction'] for s in selected['deployed_checkpoints']],axis=0)
    candidates=[]
    for weight in [0,.25,.5,.75,1]:
        p=weight*residual+(1-weight)*preds['graph_wavenet'][:n]
        candidates.append(dict(residual_weight=weight,development_mae=float(np.abs(p-truth).mean())))
    chosen=min(candidates,key=lambda s:s['development_mae'])
    freeze=dict(**chosen,candidates=candidates,baselines=rows,selection_days=list(range(126,144)),
                frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                residual_selection_sha256=digest(OUT/'selection.json'),gwn_selection_sha256=digest(V3/'selected_models.json'),
                source_sha256=digest(Path(__file__)),plan_sha256=digest(OUT.parent/'experiment_plan.json'))
    write_json(OUT/'combination_selection.json',freeze)
    confirmation=np.load(OUT/'confirmation.npz');print('confirmation keys',confirmation.files)
    residual_confirm=confirmation['prediction'];truth_confirm=confirmation['truth']
    p=chosen['residual_weight']*residual_confirm+(1-chosen['residual_weight'])*preds['graph_wavenet'][n:]
    report=dict(selected=chosen,baselines=rows,confirmation_mae=float(np.abs(p-truth_confirm).mean()),
                gwn_confirmation_mae=float(np.abs(preds['graph_wavenet'][n:]-truth_confirm).mean()))
    np.savez_compressed(OUT/'combination_confirmation.npz',truth=truth_confirm,prediction=p)
    write_json(OUT/'combination_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
