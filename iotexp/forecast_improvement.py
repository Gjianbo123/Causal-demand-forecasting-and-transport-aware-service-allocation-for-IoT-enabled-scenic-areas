"""V4 causal seasonal anchoring and learned graph residual prediction.

Only observed historical inflow, the operating clock and a fixed physical graph
enter inference. Slot templates, scaling and residual fits use training days.
This is a new method; it is not the missing original E-STGNN implementation.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Anchor:
    def __init__(self, template, scale, adjacency, first=-14, horizons=(1,2,4,6)):
        self.template=np.asarray(template,dtype=np.float32)
        self.scale=np.asarray(scale,dtype=np.float32)
        self.adjacency=np.asarray(adjacency,dtype=np.float32)
        self.first=int(first);self.horizons=np.asarray(horizons,dtype=int)

    @classmethod
    def fit(cls,data):
        # Explicit training-only slice. No validation/test observations fit a transform.
        x=np.asarray(data['inflow'][:126],dtype=np.float32)
        return cls(x.mean(0),np.maximum(x.std((0,1)),1),data['adjacency'],int(data['times'][0]))

    def features(self,raw,origin):
        raw=np.asarray(raw,dtype=np.float32);origin=np.asarray(origin,dtype=int).reshape(-1)
        if raw.ndim!=3 or raw.shape[1:]!=(12,24) or len(raw)!=len(origin):raise ValueError('Expected B x12 x24 causal history')
        times=origin[:,None]-np.arange(11,-1,-1)[None]
        if np.any(times-self.first<0):raise ValueError('Insufficient prehistory')
        past=self.template[times-self.first]
        # Poisson mean-scale estimate from the *observed* 12-frame history.
        factor=np.clip(raw.sum((1,2))/np.maximum(past.sum((1,2)),1e-6),.05,5).astype(np.float32)
        future_times=np.minimum(origin[:,None]+self.horizons[None],72)
        base=factor[:,None,None]*self.template[future_times-self.first]
        base=base.transpose(0,2,1).copy()
        residual=(raw-factor[:,None,None]*past)/self.scale[None,None]
        clock=np.stack((np.sin(2*np.pi*origin/72),np.cos(2*np.pi*origin/72)),1).astype(np.float32)
        return dict(residual=residual.astype(np.float32),base=base.astype(np.float32),factor=factor,clock=clock)

    def save(self,path):
        np.savez_compressed(path,template=self.template,scale=self.scale,adjacency=self.adjacency,
                            first=self.first,horizons=self.horizons)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:return cls(**{k:z[k] for k in z.files})


def ridge_features(features):
    r=features['residual'];a=features['factor'][:,None];c=features['clock']
    return np.concatenate((r.reshape(len(r),-1),a,c,a*c,np.ones_like(a)),1).astype(np.float64)


class GraphResidual(nn.Module):
    def __init__(self,anchor,kind='mlp',hidden=64):
        super().__init__();self.kind=kind
        adj=anchor.adjacency+np.eye(24,dtype=np.float32)
        self.register_buffer('support',torch.tensor(adj/adj.sum(1,keepdims=True)))
        self.register_buffer('scale',torch.tensor(anchor.scale))
        self.node=nn.Parameter(torch.empty(24,8));nn.init.normal_(self.node,std=.05)
        if kind=='gru':
            self.temporal=nn.GRU(3,hidden,batch_first=True)
            width=hidden+4+3+8
        else:width=12*3+4+3+8
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(.05),
                                nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,residual,base,factor,clock):
        one=torch.einsum('bln,nm->blm',residual,self.support)
        two=torch.einsum('bln,nm->blm',one,self.support)
        b=residual.shape[0]
        stack=torch.stack((residual,one,two),-1).permute(0,2,1,3)
        if self.kind=='gru':encoded=self.temporal(stack.reshape(b*24,12,3))[0][:,-1].reshape(b,24,-1)
        else:encoded=stack.reshape(b,24,-1)
        context=torch.cat((factor[:,None],clock),1)[:,None].expand(-1,24,-1)
        features=torch.cat((encoded,base/self.scale[None,:,None],context,self.node[None].expand(b,-1,-1)),-1)
        return self.head(features)


def feature_tensors(features,device):
    return {k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in features.items()}


@torch.inference_mode()
def neural_prediction(model,features,anchor,device='cpu',batch=256):
    model.eval();rows=[]
    for start in range(0,len(features['factor']),batch):
        f=feature_tensors({k:v[start:start+batch] for k,v in features.items()},device)
        residual=model(**f).cpu().numpy()
        rows.append(np.maximum(f['base'].cpu().numpy()+residual*anchor.scale[None,:,None],0))
    return np.concatenate(rows)


class ImprovedForecaster:
    """Selected single model or prespecified seed-ensemble; never fits online."""
    def __init__(self,directory,device='cpu'):
        self.directory=Path(directory);self.device=device;self.anchor=Anchor.load(self.directory/'anchor.npz')
        self.selection=json.loads((self.directory/'selection.json').read_text(encoding='utf-8'))
        self.models=[];self.coef=None
        spec=self.selection['selected']
        if spec['kind']=='ridge':self.coef=np.load(self.directory/spec['checkpoint'],allow_pickle=False)['coefficients']
        elif spec['kind'] in ('mlp','gru'):
            for item in self.selection['deployed_checkpoints']:
                saved=torch.load(self.directory/item['path'],map_location='cpu',weights_only=True)
                model=GraphResidual(self.anchor,spec['kind'],spec['hidden']).to(device)
                model.load_state_dict(saved['state_dict']);model.eval();self.models.append(model)

    def predict(self,raw,origin):
        origin=np.asarray(origin,dtype=int).reshape(-1);features=self.anchor.features(raw,origin)
        spec=self.selection['selected']
        if spec['kind']=='ridge':
            delta=(ridge_features(features)@self.coef).reshape(features['base'].shape)
            answer=np.maximum(features['base']+delta*self.anchor.scale[None,:,None],0)
        elif self.models:
            answer=np.mean([neural_prediction(m,features,self.anchor,self.device) for m in self.models],axis=0)
        else:answer=features['base']
        # Late-day control calls do not extrapolate requests beyond closing.
        return np.asarray(answer*(origin[:,None,None]+self.anchor.horizons[None,None,:]<=72),dtype=np.float32)

