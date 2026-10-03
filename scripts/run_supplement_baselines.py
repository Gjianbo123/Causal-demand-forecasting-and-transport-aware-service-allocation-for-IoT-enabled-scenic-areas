"""Transparent heuristics on the identical frozen supplementary scenarios."""
from pathlib import Path
import csv
import os
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_context import baseline_episode
from iotexp.supplement_common import V3, scenarios, make_scenario_backend, verify_v2
from iotexp.telemetry import Condition
from iotexp.util import write_json


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_v2()
    out=V3/'heuristics';out.mkdir(parents=True,exist_ok=True)
    domains={}
    for method in ('static','reactive','lookahead'):
        target=out/f'{method}_episodes.csv'
        if target.exists(): continue
        partial=target.with_suffix('.partial.csv')
        with partial.open('x',newline='',encoding='utf-8') as f:
            writer=None
            for scenario,seed,surge in scenarios():
                backend=make_scenario_backend(seed,surge)
                domains[scenario]={'seed':seed,'surge':surge,**backend.dataset_hashes()}
                for day in range(27):
                    trace=out/'traces'/f'{method}_{scenario}_d{day}.npz'
                    corrupt=int(np.random.SeedSequence([day,11,500009]).generate_state(1)[0])
                    result=baseline_episode(backend,method,day,Condition(),corrupt,trace)
                    result.pop('baseline_logic_latency_ms',None)
                    result.pop('local_pipeline_latency_ms',None)
                    row=dict(method=method,scenario=scenario,dataset_seed=seed,surge=surge,
                             environment_seed=day,training_seed=-1,trace_path=trace.relative_to(ROOT).as_posix(),**result)
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);f.flush()
                print(method,scenario,'complete',flush=True)
            os.fsync(f.fileno())
        os.replace(partial,target)
    # Re-entering a completed run must not replace its provenance with {}.
    if domains:
        write_json(out/'scenario_hashes.json',domains)


if __name__=='__main__':main()
