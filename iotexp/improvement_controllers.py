"""Additional matched-horizon comparator and static feasible reference."""
import numpy as np
from .control_supplement import FlowMPC
from .transport_rollout import TransportRollout

class SeasonalMPC(TransportRollout):
    solve=FlowMPC.solve

class StaticController(FlowMPC):
    def solve(self,inventory,queue,demand,arrivals,pending):
        h=len(demand);x=np.zeros((self.k,self.n,self.n),np.int32)
        on_site=inventory[None]+np.concatenate([np.zeros_like(arrivals[:1]),np.cumsum(arrivals,0)[:-1]],0)
        net=demand-on_site*np.asarray(self.cfg['service_per_resource'])[None,:,None]-np.asarray(self.cfg['base_service'])[None,:,None]
        q=TransportRollout.queues(queue,net)
        return dict(transfers=x,plan_transfers=np.zeros((h,*x.shape),np.int32),predicted_queue=q,
                    accepted=True,fallback=False,status=0,message='static initial allocation',optimization_ms=0.,horizon=h)
