"""Time-expanded mixed-integer receding-horizon comparator for ScenicIoT.

This NEW supplementary controller optimizes transport flows, not PPO target
scores. Its executor reuses ScenicEnvironment's service/arrival dynamics, while
replacing the nearest-source matching rule by the selected feasible flows.
No environment object or future demand is accepted by the optimization API.
"""
from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .scenic import repair_target


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 6
    time_limit_seconds: float = 0.25
    relative_gap: float = 0.01
    threads: int = 1
    presolve: bool = True


def target_scores(target):
    """Encode an integer target exactly through the original softmax repair."""
    return np.log(np.maximum(np.asarray(target, dtype=float), 1e-10))


def validate_transfers(transfers, inventory, reserved, holding):
    x = np.asarray(transfers, dtype=float)
    k, n = inventory.shape
    if x.shape != (k, n, n) or not np.isfinite(x).all():
        raise ValueError("Transfers must be finite [resource,origin,destination]")
    if (x < -1e-7).any() or np.max(np.abs(x-np.rint(x))) > 1e-5:
        raise ValueError("Transfers must be nonnegative integers")
    x = np.rint(x).astype(np.int32)
    if np.any(np.diagonal(x, axis1=1, axis2=2)):
        raise ValueError("Self-transfers are not dispatches")
    local = np.asarray(inventory)-x.sum(2)
    incoming = np.asarray(reserved)+x.sum(1)
    if (local < 0).any() or ((local+incoming) > holding).any():
        raise ValueError("Unavailable resource or destination holding violation")
    return x, local, incoming


def execute_dispatch(env, transfers):
    """Trusted actuator only; controller never receives this env.

    A dispatch is committed before the old environment's hold step. Thus a trip
    lasting tau steps loses service in precisely those tau intervals, including
    tau=1. Actual demand and FIFO accounting remain in the unchanged simulator.
    """
    begin = time.perf_counter()
    x, local, incoming = validate_transfers(transfers, env.inventory, env.reserved, env.holding)
    target = local+incoming
    _, repaired, _ = repair_target(target_scores(target), env.totals, env.holding, incoming)
    if not np.array_equal(repaired, target):
        raise RuntimeError("Exact target encoding failed")
    distance = env.data["shortest_distance_m"]
    movement = float((x*distance[None]).sum()/1000)
    new_trips = [[int(k), int(i), int(j), int(x[k,i,j]), int(env.travel_steps[k,i,j])]
                 for k,i,j in zip(*np.nonzero(x))]
    if any(t[4] < 1 for t in new_trips):
        raise ValueError("A nonlocal trip must consume at least one interval")
    env.inventory = local.astype(np.int32)
    env.reserved = incoming.astype(np.int32)
    env.trips.extend(new_trips)
    dispatch_ms = 1000*(time.perf_counter()-begin)
    frame, reward, done, audit = env.step(target_scores(target))
    if audit.movement_cost != 0:
        raise RuntimeError("Hold step unexpectedly issued an additional dispatch")
    audit.movement_cost = movement
    audit.dispatch_transfers = x
    env.last_repair_ms += dispatch_ms
    return frame, reward-env.cfg["movement_penalty"]*movement, done, audit


class FlowMPC:
    """MILP using causal queues, frozen forecasts and remembered commitments.

    Variables: start-of-interval on-site inventory a; integer departures x;
    post-interval queues q; overload o. Constraints enforce
    time-expanded resource flow, departure availability, destination reservations,
    service capacity and queue balance. Positive queue costs imply work-conserving
    service at an optimum. Predictions are conditional means, not realized demand.
    """
    def __init__(self, scenic_cfg, distance, config=None):
        self.cfg = dict(scenic_cfg)
        self.options = config or MPCConfig()
        self.distance = np.asarray(distance, dtype=float)
        self.k = len(self.cfg["resource_totals"])
        self.n = self.distance.shape[0]
        self.service_nodes = min(16, self.n)
        self.holding = np.zeros((self.k,self.n), dtype=np.int32)
        self.holding[:,:self.service_nodes] = np.asarray(self.cfg["holding_per_service_node"])[:,None]
        self.travel = np.ceil((self.distance[None]/np.asarray(self.cfg["resource_speed_m_per_min"])[:,None,None]+
                              np.asarray(self.cfg["resource_setup_minutes"])[:,None,None])*
                             self.cfg["travel_time_multiplier"]/self.cfg["decision_minutes"]).astype(int)
        self.travel[:,np.arange(self.n),np.arange(self.n)] = 0
        self.commitments = []  # own dispatched [k, origin, destination, amount, eta]
        self.templates = {}

    def reset(self):
        self.commitments = []

    def after_step(self, transfers):
        for k,i,j in zip(*np.nonzero(transfers)):
            self.commitments.append([int(k),int(i),int(j),int(transfers[k,i,j]),int(self.travel[k,i,j])])
        self.commitments = [[k,i,j,amount,eta-1] for k,i,j,amount,eta in self.commitments if eta>1]

    def resource_snapshot(self, ledger):
        size = self.k*self.n
        inventory = np.asarray(ledger[:size]).reshape(self.k,self.n)
        reserved = np.asarray(ledger[size:2*size]).reshape(self.k,self.n)
        weighted_eta = np.asarray(ledger[2*size:3*size]).reshape(self.k,self.n)
        own_reserved = np.zeros_like(reserved)
        own_eta = np.zeros_like(reserved)
        for k,_,j,amount,eta in self.commitments:
            own_reserved[k,j] += amount
            own_eta[k,j] += amount*eta
        if not np.array_equal(own_reserved,reserved) or not np.array_equal(own_eta,weighted_eta):
            raise ValueError("Trusted ledger disagrees with acknowledged dispatch history")
        return inventory.astype(int), reserved.astype(int)

    def predicted_requests(self, forecast, horizons, horizon):
        forecast = np.asarray(forecast,dtype=float)
        lead = np.arange(1,horizon+1)
        inflow = np.stack([np.interp(lead,horizons,row) for row in forecast],axis=1)
        demand = inflow[:,None,:]*np.asarray(self.cfg["request_probability"])[None,:,None]
        demand[:,:,self.service_nodes:] = 0
        return np.maximum(demand,0)

    def decide(self, window, ledger, forecast, horizons):
        horizon = min(self.options.horizon,self.cfg["episode_steps"]-int(window.decision_time))
        if horizon < 1:
            raise ValueError("No decision after closing")
        inventory,reserved = self.resource_snapshot(ledger)
        queue = np.maximum(np.asarray(window.values[-1,:,1:]).T,0)
        demand = self.predicted_requests(forecast,horizons,horizon)
        arrivals = np.zeros((horizon,self.k,self.n))
        pending = np.zeros((horizon,self.k,self.n))
        for k,_,j,amount,eta in self.commitments:
            if eta<=horizon:
                arrivals[eta-1,k,j] += amount
            pending[:min(eta,horizon),k,j] += amount
        result = self.solve(inventory,queue,demand,arrivals,pending)
        result["forecast_requests"] = demand
        return result

    def _template(self,h):
        if h in self.templates:
            return self.templates[h]
        # Hubs and gates cannot hold any resources or generate service requests;
        # eliminate their identically-zero variables without changing the model.
        k,n = self.k,self.service_nodes
        cursor = 0
        def block(shape):
            nonlocal cursor
            out = np.arange(cursor,cursor+int(np.prod(shape))).reshape(shape)
            cursor += out.size
            return out
        a,q = block((h+1,k,n)),block((h,k,n))
        o = block((h,self.service_nodes))
        arcs = [(t,c,i,j) for t in range(h) for c in range(k)
                for i in range(self.service_nodes) for j in range(self.service_nodes)
                if i!=j and t+self.travel[c,i,j]<h]
        x = np.arange(cursor,cursor+len(arcs)); cursor += len(arcs)
        outgoing,incoming,arriving = {},{},{}
        for idx,(t,c,i,j) in zip(x,arcs):
            outgoing.setdefault((t,c,i),[]).append(int(idx))
            for boundary in range(t,min(h,t+self.travel[c,i,j])):
                incoming.setdefault((boundary,c,j),[]).append(int(idx))
            arriving.setdefault((t+self.travel[c,i,j]-1,c,j),[]).append(int(idx))
        row,col,val,lower,upper,labels = [],[],[],[],[],[]
        def constraint(terms,lo,hi,label=None):
            r = len(lower)
            for c,v in terms:
                row.append(r); col.append(int(c)); val.append(float(v))
            lower.append(lo); upper.append(hi)
            if label is not None: labels.append((r,label))
        for c in range(k):
            for i in range(n):
                constraint([(a[0,c,i],1)],0,0,("initial",c,i))
        for t in range(h):
            for c in range(k):
                for i in range(n):
                    out = [(j,1) for j in outgoing.get((t,c,i),[])]
                    inc = [(j,1) for j in incoming.get((t,c,i),[])]
                    arr = [(j,-1) for j in arriving.get((t,c,i),[])]
                    constraint([(a[t+1,c,i],1),(a[t,c,i],-1)]+out+arr,0,0,("arrival",t,c,i))
                    constraint(out+[(a[t,c,i],-1)],-np.inf,0)
                    constraint([(a[t,c,i],1)]+[(j,-v) for j,v in out]+inc,
                               -np.inf,self.holding[c,i],("holding",t,c,i))
                    rate = self.cfg["service_per_resource"][c]
                    # Queue epigraph eliminates the continuous service variable:
                    # q_next >= q + arrivals - rate*(inventory-out) - base.
                    # Strictly increasing queue costs make it exact at optimum.
                    terms = [(q[t,c,i],1),(a[t,c,i],rate)]+[(j,-rate) for j,_ in out]
                    if t: terms.append((q[t-1,c,i],-1))
                    constraint(terms,0,np.inf,("queue",t,c,i))
            for i in range(self.service_nodes):
                constraint([(q[t,c,i],1) for c in range(k)]+[(o[t,i],-1)],
                           -np.inf,self.cfg["queue_capacity_per_service_node"])
        cost = np.zeros(cursor)
        cost[q.ravel()] = 1/(self.service_nodes*self.cfg["queue_reward_scale"])
        cost[o.ravel()] = self.cfg["overload_penalty"]/self.service_nodes
        for idx,(_,_,i,j) in zip(x,arcs):
            cost[idx] = self.cfg["movement_penalty"]*self.distance[i,j]/1000
        integrality = np.zeros(cursor,dtype=np.int32); integrality[x]=1
        ub = np.full(cursor,np.inf)
        ub[a.ravel()] = np.tile(self.holding[:,:n].ravel(),h+1)
        for idx,(_,c,i,_) in zip(x,arcs): ub[idx] = self.holding[c,i]
        matrix = coo_matrix((val,(row,col)),shape=(len(lower),cursor)).tocsc()
        template = dict(a=a,q=q,x=x,arcs=arcs,cost=cost,integrality=integrality,
                        ub=ub,matrix=matrix,lower=np.asarray(lower,dtype=float),upper=np.asarray(upper,dtype=float),labels=labels)
        self.templates[h] = template
        return template

    def solve(self,inventory,queue,demand,arrivals,pending):
        """Public pure optimizer API also usable for exact small-instance tests."""
        begin=time.perf_counter()
        h=len(demand); p=self._template(h)
        lower,upper=p["lower"].copy(),p["upper"].copy()
        for r,label in p["labels"]:
            kind,*indices=label
            if kind=="initial": lower[r]=upper[r]=inventory[tuple(indices)]
            elif kind=="arrival": lower[r]=upper[r]=arrivals[tuple(indices)]
            elif kind=="holding": upper[r]-=pending[tuple(indices)]
            elif kind=="queue":
                t,c,i=indices
                lower[r]=demand[t,c,i]+(queue[c,i] if t==0 else 0)-self.cfg["base_service"][c]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Unrecognized options detected.*",category=RuntimeWarning)
            result=milp(p["cost"],integrality=p["integrality"],bounds=Bounds(0,p["ub"]),
                        constraints=LinearConstraint(p["matrix"],lower,upper),
                        options={"time_limit":self.options.time_limit_seconds,
                                 "mip_rel_gap":self.options.relative_gap,"threads":self.options.threads,
                                 "presolve":self.options.presolve,"disp":False})
        transfers=np.zeros((self.k,self.n,self.n),dtype=np.int32)
        plan=np.zeros((h,self.k,self.n,self.n),dtype=np.int32)
        predicted_queue=np.full((h,self.k,self.n),np.nan)
        accepted=False; residual=None
        if result.x is not None and np.isfinite(result.x).all():
            x=np.asarray(result.x)
            ax=p["matrix"]@x
            integer_error=np.max(np.abs(x[p["x"]]-np.rint(x[p["x"]]))) if len(p["x"]) else 0
            residual=float(max(0,np.max(lower-ax),np.max(ax-upper),np.max(-x),np.max(x-p["ub"]),integer_error))
            if residual<=1e-5:
                for idx,(t,c,i,j) in zip(p["x"],p["arcs"]):
                    plan[t,c,i,j]=int(round(x[idx]))
                transfers=plan[0].copy()
                predicted_queue[:,:,:self.service_nodes]=x[p["q"]]
                predicted_queue[:,:,self.service_nodes:]=0
                validate_transfers(transfers,inventory,pending[0],self.holding)
                accepted=True
        def finite(value):
            return float(value) if value is not None and np.isfinite(value) else None
        return dict(transfers=transfers,plan_transfers=plan,predicted_queue=predicted_queue,
                    accepted=accepted,fallback=not accepted,
                    fallback_rule="hold existing positions and immutable commitments" if not accepted else "none",
                    status=int(result.status),message=str(result.message),
                    objective=finite(getattr(result,"fun",None)),
                    dual_bound=finite(getattr(result,"mip_dual_bound",None)),
                    mip_gap=finite(getattr(result,"mip_gap",None)),
                    mip_nodes=finite(getattr(result,"mip_node_count",None)),
                    primal_residual=residual,optimization_ms=1000*(time.perf_counter()-begin),
                    horizon=h,variables=len(p["cost"]),integer_variables=len(p["x"]),
                    constraints=len(lower))
