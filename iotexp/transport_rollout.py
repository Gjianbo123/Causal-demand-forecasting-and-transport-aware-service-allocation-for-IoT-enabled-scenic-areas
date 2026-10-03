"""One-dispatch marginal rollout with explicit in-transit service loss.

This is a receding-horizon heuristic, not PPO or a globally optimal planner.
Queue and overload costs preserve the original evaluation reward scale.
"""
import time
import numpy as np
from .control_supplement import FlowMPC,validate_transfers

class TransportRollout(FlowMPC):
    def __init__(self,scenic_cfg,distance,config,anchor,terminal_weight=0):
        super().__init__(scenic_cfg,distance,config)
        self.anchor=anchor;self.terminal_weight=float(terminal_weight)

    def decide(self,window,ledger,forecast,horizons):
        self.origin=int(window.decision_time)
        return super().decide(window,ledger,forecast,horizons)

    def predicted_requests(self,forecast,horizons,horizon):
        demand=super().predicted_requests(forecast,horizons,horizon)
        last=int(max(horizons))
        if horizon>last:
            reference=self.anchor.template[min(72,self.origin+last)-self.anchor.first]
            scale=np.clip(np.asarray(forecast)[:,-1]/np.maximum(reference,.25),.05,5)
            tail=self.anchor.template[self.origin+np.arange(last+1,horizon+1)-self.anchor.first]*scale[None]
            demand[last:]=tail[:,None,:]*np.asarray(self.cfg['request_probability'])[None,:,None]
            demand[:,:,self.service_nodes:]=0
        return demand

    @staticmethod
    def queues(initial,net):
        raw=initial[None]+np.cumsum(net,axis=0)
        return raw-np.minimum(0,np.minimum.accumulate(raw,axis=0))

    def node_cost(self,q):
        return (q.sum((0,1))+self.terminal_weight*q[-1].sum(0))/(self.service_nodes*self.cfg['queue_reward_scale'])+self.cfg['overload_penalty']*np.maximum(q.sum(1)-self.cfg['queue_capacity_per_service_node'],0).sum(0)/self.service_nodes

    def solve(self,inventory,queue,demand,arrivals,pending):
        started=time.perf_counter();h=len(demand);n=self.service_nodes
        rate=np.asarray(self.cfg['service_per_resource'])[:,None]
        # Arrival at end of an interval becomes productive only next interval.
        on_site=inventory[None]+np.concatenate([np.zeros_like(arrivals[:1]),np.cumsum(arrivals,axis=0)[:-1]],axis=0)
        capacity=on_site*rate[None]+np.asarray(self.cfg['base_service'])[None,:,None]
        net=demand-capacity
        x=np.zeros((self.k,self.n,self.n),dtype=np.int32)
        remaining=inventory.copy();occupancy=inventory+pending[0]
        steps=0
        while True:
            q=self.queues(queue,net);cost=self.node_cost(q)
            best=(0.,None)
            for k in range(self.k):
                removable=remaining[k,:n]>0;space=occupancy[k,:n]<self.holding[k,:n]
                if not removable.any() or not space.any():continue
                removed=q.copy();removed[:,k]=self.queues(queue[k],net[:,k]+rate[k,0])
                loss=self.node_cost(removed)-cost
                gains={}
                for tau in np.unique(self.travel[k,:n,:n]):
                    if tau<1 or tau>=h:continue
                    change=np.zeros_like(net[:,k]);change[int(tau):]=-rate[k,0]
                    added=q.copy();added[:,k]=self.queues(queue[k],net[:,k]+change)
                    gains[int(tau)]=cost-self.node_cost(added)
                benefit=np.full((n,n),-np.inf)
                for tau,gain in gains.items():
                    mask=(self.travel[k,:n,:n]==tau)&removable[:,None]&space[None]
                    score=gain[None,:n]-loss[:n,None]-self.cfg['movement_penalty']*self.distance[:n,:n]/1000
                    benefit[mask]=score[mask]
                flat=int(np.argmax(benefit));i,j=np.unravel_index(flat,benefit.shape)
                if benefit[i,j]>best[0]+1e-10:best=(float(benefit[i,j]),(k,i,j))
            if best[1] is None:break
            k,i,j=best[1];tau=int(self.travel[k,i,j]);x[k,i,j]+=1
            remaining[k,i]-=1;occupancy[k,i]-=1;occupancy[k,j]+=1
            net[:,k,i]+=rate[k,0];net[tau:,k,j]-=rate[k,0];steps+=1
            if steps>int(inventory.sum()):raise RuntimeError('Cannot dispatch newly arriving units')
        validate_transfers(x,inventory,pending[0],self.holding)
        plan=np.zeros((h,self.k,self.n,self.n),dtype=np.int32);plan[0]=x
        q=self.queues(queue,net)
        return dict(transfers=x,plan_transfers=plan,predicted_queue=q,accepted=True,fallback=False,
                    status=0,message='greedy marginal rollout; no global optimality certificate',
                    objective=float(self.node_cost(q).sum()+self.cfg['movement_penalty']*(x*self.distance[None]).sum()/1000),
                    optimization_ms=1000*(time.perf_counter()-started),horizon=h,greedy_moves=steps)
