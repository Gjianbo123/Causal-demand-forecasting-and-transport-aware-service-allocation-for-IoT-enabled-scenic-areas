"""Frozen causal prediction adapters for v4; no environment/future-label input."""
from pathlib import Path
import json
import numpy as np
import torch
from .forecast_improvement import ImprovedForecaster
from .forecast_supplement import make_model
from .scenic_forecast import _clock
from .telemetry import TrainingStats
from .supplement_common import SupplementBackend,ROOT,V3

V4=ROOT/'runs/improvement_v4'

class ForecastRuntime:
    def __init__(self,dataset,name='combined',device='cpu'):
        self.name=name;self.device=device;self.models=[]
        self.stats=TrainingStats.load(V3/'forecast/training_stats.npz')
        self.residual=ImprovedForecaster(V4/'forecast',device) if name in ['combined','residual'] else None
        self.weight=json.loads((V4/'forecast/combination_selection.json').read_text())['residual_weight']
        if name not in ['residual','anchor']:
            family='graph_wavenet' if name=='combined' else name
            for spec in json.loads((V3/'forecast/selected_models.json').read_text()):
                if spec['model']!=family:continue
                saved=torch.load(V3/'forecast'/spec['checkpoint'],map_location='cpu',weights_only=True)
                model=make_model(family,spec['candidate'],dataset,dict(horizons=[1,2,4,6],history=12,period=72)).to(device)
                model.load_state_dict(saved['state_dict']);model.eval();self.models.append(model)
        if not self.models and not self.residual:raise ValueError(name)

    @torch.inference_mode()
    def predict(self,raw,origin,batch=256):
        raw=np.asarray(raw,dtype=np.float32);origin=np.asarray(origin,dtype=int)
        if self.models:
            mean=self.stats.mean[:,0].astype(np.float32);scale=self.stats.scale[:,0].astype(np.float32)
            answers=[]
            for start in range(0,len(raw),batch):
                r=raw[start:start+batch];o=origin[start:start+batch]
                x=torch.as_tensor((r-mean)/scale,device=self.device)
                clock=torch.as_tensor(_clock(o[:,None]-np.arange(11,-1,-1)[None],72),device=self.device)
                ot=torch.as_tensor(o,device=self.device)
                # Clip each model's raw prediction before averaging, matching v3.
                ps=[np.maximum(m(x,clock,ot).cpu().numpy()*scale[None,:,None]+mean[None,:,None],0) for m in self.models]
                answers.append(np.mean(ps,axis=0))
            p=np.concatenate(answers)
        if self.residual:
            r=self.residual.predict(raw,origin)
            p=r if not self.models else self.weight*r+(1-self.weight)*p
        return (p*(origin[:,None,None]+np.array([1,2,4,6])[None,None,:]<=72)).astype(np.float32)

class ImprovedBackend(SupplementBackend):
    def __init__(self,config,predictor='combined',runtime=None,cache=True):
        super().__init__(config,'edge_stgru',cache_forecasts=cache)
        self.runtime=runtime or ForecastRuntime(self.data,predictor)
        self.predictor_name=predictor

    def forecast(self,window):
        raw=np.asarray(window.values[-12:,:,0],dtype=np.float32)
        key=(int(window.decision_time),raw.tobytes())
        if not self.cache_forecasts:return self.runtime.predict(raw[None],[key[0]])[0]
        if key not in self._forecast_cache:self._forecast_cache[key]=self.runtime.predict(raw[None],[key[0]])[0]
        return self._forecast_cache[key].copy()

    def prime_causal_cache(self,physical_days):
        # Batching only evaluates a pure function of each complete input key.
        # No future labels enter the predictor. Do not use for latency reporting.
        positions=np.arange(14,86);hist=positions[:,None]-np.arange(11,-1,-1)[None]
        raw=self.data['inflow'][physical_days][:,hist].reshape(-1,12,24).astype(np.float32)
        origins=np.tile(np.arange(72),len(physical_days));p=self.runtime.predict(raw,origins)
        for r,o,pred in zip(raw,origins,p):self._forecast_cache[(int(o),r.tobytes())]=pred.copy()
