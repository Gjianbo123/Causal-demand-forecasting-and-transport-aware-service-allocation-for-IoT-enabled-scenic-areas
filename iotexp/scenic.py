"""ScenicIoT-Rebuild-v1: a NEW, fully synthetic, disclosed service-request simulator.

This is not ScenicFlow-Sim, a reconstruction of unavailable source, or a digital
twin calibrated to a real site. All unspecified parameters are new assumptions.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .contracts import Frame, Audit
from .demo import largest_remainder


DEFAULT_SCENIC = {
    "name": "ScenicIoT-Rebuild-v1", "dataset_seed": 20260929,
    "days": 180, "split_days": [126, 27, 27], "nodes": 24, "warmup_extra": 2,
    "history": 12, "horizons": [1, 2, 4, 6], "episode_steps": 72,
    "decision_minutes": 10, "resource_totals": [30, 24, 16, 12],
    "holding_per_service_node": [5, 4, 3, 3],
    "service_per_resource": [2, 2, 2, 2], "base_service": [0, 0, 0, 0],
    "request_probability": [0.75, 0.55, 0.40, 0.30],
    "resource_speed_m_per_min": [80, 55, 40, 30],
    "resource_setup_minutes": [2, 4, 6, 8], "travel_time_multiplier": 1.0,
    "queue_capacity_per_service_node": 50,
    "gate_base": [5, 6, 5, 6], "gate_peak": [18, 21, 19, 22],
    "peak_centers": [20, 47], "peak_widths": [8, 10],
    "afternoon_peak_ratio": 0.8, "gate_phase_offsets": [-4, 0, 4, 8],
    "graph_propagation": 0.68, "intensity_log_sd": 0.08,
    "day_log_sd": 0.12, "weekend_multiplier": 1.2,
    "holiday_multiplier": 1.35, "holiday_days": [17,18,47,48,77,78,107,108,137,138,167,168],
    "weather_values": [0.65, 0.9, 1.0, 1.1],
    "weather_probabilities": [0.15, 0.2, 0.45, 0.2],
    "event_probability": 0.3, "event_amplitude": 18, "event_width": 7,
    "movement_penalty": 0.025, "overload_penalty": 0.02,
    "queue_reward_scale": 25.0,
}


def scenic_config(config):
    supplied = config.get("scenic", config)
    result = dict(DEFAULT_SCENIC)
    result.update(supplied)
    if result["nodes"] != 24 or len(result["resource_totals"]) != 4:
        raise ValueError("ScenicIoT-Rebuild-v1 requires 24 nodes and four resource types")
    if sum(result["split_days"]) != result["days"] or min(result["split_days"]) < 1:
        raise ValueError("Day partitions must be positive and sum to days")
    if not 0 <= result["graph_propagation"] < 1:
        raise ValueError("Graph propagation must be in [0,1)")
    if min(result["resource_speed_m_per_min"]) <= 0 or result["travel_time_multiplier"] <= 0:
        raise ValueError("Travel speed and multiplier must be positive")
    if any(t > 16*c for t,c in zip(result["resource_totals"], result["holding_per_service_node"])):
        raise ValueError("Insufficient service-node resource holding capacity")
    return result


def scenic_graph():
    """Undirected 31-edge connected graph; lengths in metres, no edge congestion."""
    edges, lengths = [], []
    for cluster in range(4):
        for local in range(4):
            edges.append((4*cluster+local, 16+cluster))
            lengths.append(100 + 30*local + 15*cluster)
        edges.append((16+cluster, 20+cluster))
        lengths.append(140 + 20*cluster)
    for a,b,length in ((16,17,260),(17,18,300),(18,19,280),(19,16,320),
                       (0,1,90),(2,3,120),(4,5,100),(6,7,130),
                       (8,9,110),(10,11,140),(12,13,120)):
        edges.append((a,b)); lengths.append(length)
    adjacency = np.zeros((24,24), dtype=np.float32)
    distance = np.full((24,24), np.inf)
    np.fill_diagonal(distance, 0)
    for (a,b), length in zip(edges,lengths):
        adjacency[a,b] = adjacency[b,a] = 1.0
        distance[a,b] = distance[b,a] = float(length)
    for k in range(24):
        distance = np.minimum(distance, distance[:,k,None] + distance[None,k,:])
    return {"adjacency": adjacency, "edge_index": np.asarray(edges,dtype=np.int32),
            "edge_lengths_m": np.asarray(lengths,dtype=np.float32),
            "shortest_distance_m": distance.astype(np.float32),
            "node_types": np.asarray(["service"]*16+["hub"]*4+["gate"]*4)}


def uniform_allocation(cfg):
    allocation = np.zeros((4,24),dtype=np.int32)
    for k,total in enumerate(cfg["resource_totals"]):
        allocation[k,:16] = largest_remainder(np.ones(16),total)
    return allocation


def build_dataset(config):
    """Return deterministic days x (history+2+72+1) x nodes exogenous visit counts.

    ``times`` is -(history+2),...,72. Environment arrivals begin at t=1; t<=0
    provides an observed pre-opening inflow history but no carry-in requests.
    ``observations`` adds four STATIC-policy queue channels for fitting train-only
    preprocessing; they are never substituted for a controlled rollout's queues.
    """
    cfg = scenic_config(config)
    graph = scenic_graph()
    rng = np.random.default_rng(cfg["dataset_seed"])
    days, history, steps = cfg["days"], cfg["history"], cfg["episode_steps"]
    warmup = history + cfg["warmup_extra"]
    times = np.arange(-warmup,steps+1)
    day_ids = np.arange(days)
    weekend = (day_ids % 7 >= 5)
    holiday = np.isin(day_ids,cfg["holiday_days"])
    weather = rng.choice(cfg["weather_values"],size=days,p=cfg["weather_probabilities"])
    daily = np.exp(rng.normal(-cfg["day_log_sd"]**2/2,cfg["day_log_sd"],days))
    daily *= np.where(weekend,cfg["weekend_multiplier"],1)
    daily *= np.where(holiday,cfg["holiday_multiplier"],1)*weather
    event_active = rng.random(days) < cfg["event_probability"]
    event_node = rng.integers(0,16,days)
    event_center = rng.integers(25,55,days)
    # Known exogenous conditions, not a future realized demand series.
    day_features = np.column_stack((day_ids%7,holiday,weather,event_active,event_node)).astype(np.float32)
    transition = graph["adjacency"] / graph["adjacency"].sum(axis=1,keepdims=True)
    # Each source node distributes expected visit events uniformly over neighbors.
    intensity = np.zeros((days,24),dtype=np.float64)
    inflow = np.zeros((days,len(times),24),dtype=np.int32)
    gate_phase = np.asarray(cfg["gate_phase_offsets"])
    for j,t in enumerate(times):
        shifted = t - gate_phase
        peak = np.exp(-0.5*((shifted-cfg["peak_centers"][0])/cfg["peak_widths"][0])**2)
        peak += cfg["afternoon_peak_ratio"]*np.exp(-0.5*((shifted-cfg["peak_centers"][1])/cfg["peak_widths"][1])**2)
        external = np.zeros((days,24))
        external[:,20:] = daily[:,None]*(np.asarray(cfg["gate_base"])+np.asarray(cfg["gate_peak"])*peak)
        external[day_ids,event_node] += (daily*event_active*cfg["event_amplitude"]*
                                        np.exp(-0.5*((t-event_center)/cfg["event_width"])**2))
        intensity = external + cfg["graph_propagation"]*(intensity @ transition)
        perturbation = np.exp(rng.normal(-cfg["intensity_log_sd"]**2/2,cfg["intensity_log_sd"],intensity.shape))
        inflow[:,j] = rng.poisson(intensity*perturbation)
    demand = np.zeros((days,len(times),4,24),dtype=np.int32)
    for k,p in enumerate(cfg["request_probability"]):
        demand[:,:,k,:16] = rng.binomial(inflow[:,:,:16],p)
    observations = np.zeros((days,len(times),24,5),dtype=np.float32)
    observations[:,:,:,0] = inflow
    static = uniform_allocation(cfg)
    service = static*np.asarray(cfg["service_per_resource"])[:,None]
    service[:,:16] += np.asarray(cfg["base_service"])[:,None]
    queue = np.zeros((days,4,24),dtype=np.int32)
    for j,t in enumerate(times):
        if t > 0:
            queue = np.maximum(queue + demand[:,j] - service,0)
        observations[:,j,:,1:] = queue.transpose(0,2,1)
    return {**graph,"inflow":inflow,"demand":demand,"observations":observations,
            "times":times,"day_features":day_features,"day_ids":day_ids,
            "split_days":np.asarray(cfg["split_days"],dtype=np.int32),"config":cfg}


def repair_target(raw, totals, holding, reserved):
    """Forecast-independent target repair; committed arrivals are lower bounds."""
    raw = np.asarray(raw,dtype=float)
    if raw.shape != holding.shape or raw.shape != reserved.shape or not np.isfinite(raw).all():
        raise ValueError("Raw action must be finite [4,24]")
    candidate = np.zeros_like(holding)
    executed = np.zeros_like(holding)
    for k,total in enumerate(totals):
        permitted = np.flatnonzero(holding[k] > 0)
        weights = np.exp(raw[k,permitted]-raw[k,permitted].max())
        candidate[k,permitted] = largest_remainder(weights,int(total))
        executed[k] = np.maximum(reserved[k],np.minimum(candidate[k],holding[k]))
        difference = int(total-executed[k].sum())
        if difference > 0:
            order = permitted[np.lexsort((permitted,-raw[k,permitted]))]
            for node in order:
                added = min(difference,int(holding[k,node]-executed[k,node]))
                executed[k,node] += added; difference -= added
        elif difference < 0:
            order = permitted[np.lexsort((permitted,raw[k,permitted]))]
            for node in order:
                removed = min(-difference,int(executed[k,node]-reserved[k,node]))
                executed[k,node] -= removed; difference += removed
        if difference:
            raise RuntimeError("Resource reservations cannot be repaired within feasible bounds")
    feasible = lambda a: bool((a>=reserved).all() and (a<=holding).all() and
                              np.array_equal(a.sum(1),totals))
    return candidate,executed,not feasible(candidate)


class ScenicEnvironment:
    """Daily terminal task with FIFO *service requests*, not distinct visitors."""
    def __init__(self,cfg,dataset,split,seed):
        self.cfg,self.data,self.split,self.seed = cfg,dataset,split,int(seed)
        counts = cfg["split_days"]
        index = {"train":0,"validation":1,"test":2}[split]
        self.day = sum(counts[:index]) + self.seed % counts[index]
        self.n,self.k,self.history,self.steps = 24,4,cfg["history"],cfg["episode_steps"]
        self.warmup = self.history + cfg["warmup_extra"]
        self.totals = np.asarray(cfg["resource_totals"],dtype=np.int32)
        self.holding = np.zeros((4,24),dtype=np.int32)
        self.holding[:,:16] = np.asarray(cfg["holding_per_service_node"])[:,None]
        self.capacities = np.zeros(24)
        self.capacities[:16] = cfg["queue_capacity_per_service_node"]
        self.overload_nodes = np.arange(24)<16
        travel = (dataset["shortest_distance_m"][None,:,:] /
                  np.asarray(cfg["resource_speed_m_per_min"])[:,None,None] +
                  np.asarray(cfg["resource_setup_minutes"])[:,None,None])
        self.travel_steps = np.ceil(travel*cfg["travel_time_multiplier"]/cfg["decision_minutes"]).astype(int)
        self.travel_steps[:,np.arange(24),np.arange(24)] = 0
        self.service_rate = np.asarray(cfg["service_per_resource"],dtype=np.int32)[:,None]
        self.base_service = np.zeros((4,24),dtype=np.int32)
        self.base_service[:,:16] = np.asarray(cfg["base_service"],dtype=np.int32)[:,None]

    def reset(self):
        self.t = 0
        self.inventory = uniform_allocation(self.cfg)
        self.reserved = np.zeros((4,24),dtype=np.int32)
        # [resource,origin,destination,quantity,remaining_intervals]
        self.trips = []
        self.queue_batches = np.zeros((4,24,self.steps+1),dtype=np.int32)
        self.queue = np.zeros((4,24),dtype=np.int32)
        self.arrived_requests = self.served_requests = 0
        self.last_repair_ms = 0.0
        return [Frame(t,self._observation(t),self._ledger()) for t in range(-self.warmup,1)]

    def _observation(self,t):
        inflow = self.data["inflow"][self.day,self.warmup+t]
        return np.column_stack((inflow,self.queue.T)).astype(np.float32)

    def _ledger(self):
        remaining = np.zeros((4,24),dtype=np.float32)
        for k,origin,destination,quantity,eta in self.trips:
            remaining[k,destination] += quantity*eta
        return np.concatenate((self.inventory.ravel(),self.reserved.ravel(),remaining.ravel(),
                               self.data["day_features"][self.day])).astype(np.float32)

    def _assert_conservation(self):
        if not np.array_equal((self.inventory+self.reserved).sum(1),self.totals):
            raise RuntimeError("Resource conservation failed")
        if (self.inventory<0).any() or (self.reserved<0).any() or ((self.inventory+self.reserved)>self.holding).any():
            raise RuntimeError("Negative inventory or destination reservation exceeded capacity")
        if int(self.queue.sum())+self.served_requests != self.arrived_requests:
            raise RuntimeError("Service request conservation failed")

    def step(self,raw_action):
        if self.t >= self.steps:
            raise RuntimeError("Episode already terminated")
        raw_action = np.asarray(raw_action)
        if raw_action.shape == (4, 16):
            # Policies sample only scores for the 16 admissible service nodes.
            # Hub/gate scores have no action semantics and no nuisance densities.
            raw_action = np.pad(raw_action, ((0, 0), (0, 8)), constant_values=0)
        # Exactly the waiting accrued during (t,t+1]; terminal requests are censored.
        queue_area_min = float(self.queue.sum()*self.cfg["decision_minutes"])
        repair_started = time.perf_counter()
        candidate,target,pre = repair_target(raw_action,self.totals,self.holding,self.reserved)
        current = self.inventory+self.reserved
        movement = 0.0
        for k in range(4):
            surplus = np.maximum(current[k]-target[k],0)
            deficit = np.maximum(target[k]-current[k],0)
            for destination in np.flatnonzero(deficit):
                origins = np.flatnonzero(surplus)
                order = origins[np.lexsort((origins,self.data["shortest_distance_m"][origins,destination]))]
                for origin in order:
                    quantity = min(int(surplus[origin]),int(deficit[destination]))
                    if not quantity:
                        continue
                    self.inventory[k,origin] -= quantity
                    self.reserved[k,destination] += quantity
                    surplus[origin] -= quantity; deficit[destination] -= quantity
                    self.trips.append([k,int(origin),int(destination),quantity,int(self.travel_steps[k,origin,destination])])
                    movement += quantity*float(self.data["shortest_distance_m"][origin,destination])/1000
                if deficit[destination]:
                    raise RuntimeError("Not enough local inventory for feasible target")
        # Deliberately excludes request generation, FIFO service and queue scoring.
        self.last_repair_ms = 1000*(time.perf_counter()-repair_started)
        self.t += 1
        requests = self.data["demand"][self.day,self.warmup+self.t]
        self.queue_batches[:,:,self.t] += requests
        self.queue += requests
        self.arrived_requests += int(requests.sum())
        # A resource must remain locally present throughout this service interval.
        capacity = self.inventory*self.service_rate+self.base_service
        served = np.minimum(capacity,self.queue)
        waits = []
        for k,node in zip(*np.nonzero(served)):
            count = int(served[k,node])
            self.queue[k,node] -= count
            self.served_requests += count
            for arrived in np.flatnonzero(self.queue_batches[k,node,:self.t+1]):
                take = min(count,int(self.queue_batches[k,node,arrived]))
                self.queue_batches[k,node,arrived] -= take
                waits.extend([(self.t-int(arrived))*self.cfg["decision_minutes"]]*take)
                count -= take
                if count == 0:
                    break
        # Arrivals at the end of this interval are first eligible in the next one.
        in_transit = []
        for trip in self.trips:
            trip[4] -= 1
            k,origin,destination,quantity,eta = trip
            if eta <= 0:
                self.reserved[k,destination] -= quantity
                self.inventory[k,destination] += quantity
            else:
                in_transit.append(trip)
        self.trips = in_transit
        self._assert_conservation()
        occupancy = self.queue.sum(0)
        excess = np.maximum(occupancy[:16]-self.capacities[:16],0).mean()
        reward = -(occupancy[:16].mean()/self.cfg["queue_reward_scale"] +
                   self.cfg["movement_penalty"]*movement+self.cfg["overload_penalty"]*excess)
        audit = Audit(candidate,target,pre,False,not np.array_equal(candidate,target),
                      occupancy.copy(),self.capacities.copy(),self.overload_nodes.copy(),
                      np.asarray(waits,dtype=np.float32),int(self.queue.sum()),int(requests.sum()),movement)
        audit.queue_area_min = queue_area_min
        audit.queue_post_area_min = float(self.queue.sum()*self.cfg["decision_minutes"])
        audit.resource_in_transit_units = int(self.reserved.sum())
        audit.mean_queue = float(occupancy[:16].mean())
        return Frame(self.t,self._observation(self.t),self._ledger()),float(reward),self.t==self.steps,audit


class ScenicBackend:
    """Formal NEW synthetic experiment, explicitly not a restored original backend."""
    is_demo = False
    def __init__(self,config):
        self.config = config
        self.cfg = scenic_config(config)
        self.nodes,self.resources = 24,4
        self.action_shape = (4, 16)
        self.history,self.horizons = self.cfg["history"],self.cfg["horizons"]
        self.state_dim = self.history*self.nodes*7 + 3*self.resources*self.nodes + 5 + 2
        self.data = build_dataset(self.cfg)
        self.stats,self.forecaster = None,None
        scale = np.repeat(np.asarray(self.cfg["resource_totals"]),self.nodes)
        max_eta = max(1,np.ceil((self.data["shortest_distance_m"].max()/min(self.cfg["resource_speed_m_per_min"])+
                               max(self.cfg["resource_setup_minutes"]))*self.cfg["travel_time_multiplier"]/self.cfg["decision_minutes"]))
        self.ledger_scale = np.concatenate((scale,scale,scale*max_eta,[6,1,1.1,1,15])).astype(np.float32)

    def make_env(self,split,seed):
        return ScenicEnvironment(self.cfg,self.data,split,seed)

    def prepare(self,output_dir):
        from .scenic_forecast import prepare_forecaster
        output = Path(output_dir)
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/"synthetic_dataset.npz",**{k:v for k,v in self.data.items() if k!="config"})
        (output/"simulator_config.json").write_text(json.dumps(self.cfg,indent=2),encoding="utf-8")
        self.stats,self.forecaster = prepare_forecaster(self.data,self.config,output)

    def load(self,output_dir):
        from .scenic_forecast import load_forecaster
        output = Path(output_dir)
        saved = json.loads((output/"simulator_config.json").read_text(encoding="utf-8"))
        if saved != self.cfg:
            raise ValueError("Simulator configuration differs from frozen dataset")
        with np.load(output/"synthetic_dataset.npz",allow_pickle=False) as arrays:
            for key in arrays.files:
                if key in self.data and not np.array_equal(arrays[key],self.data[key]):
                    raise ValueError(f"Rebuilt dataset mismatch: {key}")
        self.stats,self.forecaster = load_forecaster(output)

    def forecast(self,window):
        from .scenic_forecast import forecast_window
        prediction = forecast_window(self.forecaster,window,self.stats,self.config)
        # Public operating schedule, not future simulator values: requests stop
        # at closing and predictions beyond that finite episode have no target.
        after_closing = window.decision_time + np.asarray(self.horizons) > self.cfg["episode_steps"]
        prediction[:,after_closing] = 0.0
        return prediction

    def encode_state(self,window,ledger):
        phase = 2*np.pi*window.decision_time/self.cfg["episode_steps"]
        return np.concatenate((self.stats.normalize(window.values).ravel(),window.mask.ravel(),
                               (window.age/self.history).ravel(),ledger/self.ledger_scale,
                               [np.sin(phase),np.cos(phase)])).astype(np.float32)

    def heuristic_action(self,window,mode):
        """Fixed transparent comparators; only window and frozen forecasts are read."""
        if mode.lower() in ("static","uniform"):
            return np.zeros((4,24),dtype=np.float32)
        observed = np.maximum(window.values[-1],0)
        queue = observed[:,1:].T
        probability = np.asarray(self.cfg["request_probability"])[:,None]
        if mode.lower() in ("reactive","reactive_demand"):
            expected = probability*observed[None,:,0]
        elif mode.lower() in ("lookahead","forecast","forecast_lookahead"):
            forecast = self.forecast(window)
            distance = self.data["shortest_distance_m"][:16,:16]
            positive = distance[distance>0]
            leads = np.ceil((np.median(positive)/np.asarray(self.cfg["resource_speed_m_per_min"])+
                             np.asarray(self.cfg["resource_setup_minutes"]))*
                            self.cfg["travel_time_multiplier"]/self.cfg["decision_minutes"])
            indices = [int(np.argmin(np.abs(np.asarray(self.horizons)-lead))) for lead in leads]
            expected = probability*np.stack([forecast[:,j] for j in indices])
        else:
            raise ValueError(f"Unknown heuristic mode: {mode}")
        weights = (queue+expected)/np.asarray(self.cfg["service_per_resource"])[:,None] + 0.1
        return np.log(np.maximum(weights,1e-6)).astype(np.float32)

    def describe(self):
        return {"backend":"ScenicIoT-Rebuild-v1","is_demo":False,"reproduces_original":False,
                "data_provenance":"NEW fully synthetic uncalibrated graph-propagated visit-event and service-request model",
                "metric_population":"independent service requests; not unique tourists or complete visitor itineraries",
                "split":"fixed chronological synthetic days: 0-125 train, 126-152 validation, 153-179 test",
                "seed_semantics":"day = split_start + environment_seed modulo split_day_count",
                "waiting":"FIFO completed service requests, complete 10-minute bins; terminal backlog separately reported",
                "occupancy":"sum of four service-request queues at 16 service nodes, NOT tourist occupancy",
                "movement_cost_unit":"resource-kilometres",
                "repair":"raw-score target allocation; holding caps and immutable incoming reservations; nearest-source dispatch",
                "travel":"shortest undirected graph distance / type speed + setup, rounded up to 10-minute intervals",
                "terminal":"finite operating day; remaining requests unserved, in-transit units still conserved",
                "forecast":"preselected edge_stgru seed0; epoch selected on validation; inspect forecast/selected.json",
                "inflow_sha256":hashlib.sha256(self.data["inflow"].tobytes()).hexdigest(),
                "graph_edges":self.data["edge_index"].tolist(),"edge_lengths_m":self.data["edge_lengths_m"].tolist(),
                "configuration":self.cfg}
