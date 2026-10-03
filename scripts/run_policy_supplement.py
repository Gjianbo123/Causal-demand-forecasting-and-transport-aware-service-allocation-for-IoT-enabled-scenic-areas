"""Longer-budget and predictor-swap experiments under a frozen protocol."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def config_for(predictor):
    from iotexp.supplement_common import base_config
    config = base_config()
    config['ppo']['total_steps'] = 43200
    config['methods'] = ['F0', 'F1', 'F4'] if predictor == 'edge_stgru' else ['F1', 'F4']
    config['trace_mode'] = 'compact'
    return config


def worker(args):
    import numpy as np
    import torch
    from iotexp.engine import train_one, make_policy, evaluate_episode
    from iotexp.supplement_common import V2, V3, load_base_backend, make_scenario_backend, scenarios, verify_v2
    from iotexp.telemetry import Condition
    from iotexp.util import digest, write_json
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_v2()
    config = config_for(args.predictor)
    directory = V3 / 'policies' / args.predictor
    if args.stage == 'train':
        backend = load_base_backend(args.predictor)
        started = time.perf_counter()
        summary = train_one(backend, config, args.method, args.seed, directory)
        summary['elapsed_seconds'] = time.perf_counter() - started
        write_json(directory / 'checkpoints' / f'{args.method}_seed{args.seed}' / 'supplement_metadata.json', summary)
        print(json.dumps(summary), flush=True)
        return
    checkpoint_dir = (V2 if args.budget == 'legacy' else directory) / 'checkpoints' / f'{args.method}_seed{args.seed}'
    checkpoint = checkpoint_dir / 'best.pt'
    metadata = json.loads((checkpoint_dir / 'metadata.json').read_text())
    if digest(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint modified')
    output = directory / 'evaluation_shards' / f'{args.method}_seed{args.seed}_{args.budget}.csv'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.csv')
    with partial.open('x', newline='', encoding='utf-8') as f:
        writer = None
        for scenario, data_seed, surge in scenarios():
            backend = make_scenario_backend(data_seed, surge, args.predictor)
            model = make_policy(backend, config, args.method, args.seed)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
            for day in range(27):
                trace = directory / 'traces' / f'{args.method}_s{args.seed}_{args.budget}_{scenario}_d{day}.npz'
                corruption = int(np.random.SeedSequence([day, 11, 500009]).generate_state(1)[0])
                result = evaluate_episode(model, backend, config, 'test', day, Condition(), corruption, trace)
                # Cached forecasting invalidates end-to-end timing; do not report it.
                result.pop('local_pipeline_latency_ms', None)
                result.pop('policy_latency_ms', None)
                row = dict(method=args.method, predictor=args.predictor, training_seed=args.seed,
                           budget_steps=14400 if args.budget == 'legacy' else 43200,
                           best_step=metadata['best_step'], scenario=scenario, dataset_seed=data_seed,
                           surge=surge, environment_seed=day, checkpoint_sha256=digest(checkpoint),
                           trace_path=trace.relative_to(ROOT).as_posix(), **result)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); f.flush()
            print(f'{args.predictor} {args.method} {args.seed} {args.budget}: {scenario}', flush=True)
        os.fsync(f.fileno())
    os.replace(partial, output)


def orchestrate(args):
    from iotexp.supplement_common import V2, V3, verify_v2
    from iotexp.util import digest, write_json
    verify_v2()
    stage = args.stage.removeprefix('parallel-')
    logdir = V3 / 'policies/logs'; logdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in range(10):
        for predictor in ('edge_stgru', 'ridge'):
            methods = ['F0','F1','F4'] if predictor == 'edge_stgru' else ['F1','F4']
            for method in methods:
                tasks.append((predictor, method, seed, 'long'))
        if stage == 'evaluate':
            tasks += [('edge_stgru', method, seed, 'legacy') for method in ('F0','F1','F4')]
    for predictor in ('edge_stgru','ridge'):
        cfg = V3 / 'policies' / predictor / 'config.json'
        if not cfg.exists(): write_json(cfg, config_for(predictor))
    running = []
    while tasks or running:
        for task in list(tasks):
            if len(running) >= args.workers: break
            predictor, method, seed, budget = task
            directory = V3/'policies'/predictor
            target = directory / ('checkpoints/'+f'{method}_seed{seed}/metadata.json' if stage == 'train'
                                   else 'evaluation_shards/'+f'{method}_seed{seed}_{budget}.csv')
            if target.exists(): tasks.remove(task); continue
            checkpoint_dir = (V2 if budget == 'legacy' else directory) / 'checkpoints' / f'{method}_seed{seed}'
            if stage == 'evaluate' and not (checkpoint_dir/'metadata.json').exists(): continue
            path = logdir/f'{stage}_{predictor}_{method}_{seed}_{budget}.log'
            log = path.open('w',encoding='utf-8')
            cmd=[sys.executable,str(Path(__file__).resolve()),stage,'--predictor',predictor,'--method',method,'--seed',str(seed),'--budget',budget]
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            running.append((p,log,task)); tasks.remove(task)
            print('Started',stage,task,flush=True)
        for item in list(running):
            p,log,task=item
            if p.poll() is not None:
                log.close();running.remove(item)
                if p.returncode:
                    for other,fh,_ in running: other.terminate();fh.close()
                    raise RuntimeError(f'Failed {stage} {task}; inspect logs')
                print('Completed',stage,task,flush=True)
        time.sleep(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','evaluate','parallel-train','parallel-evaluate'])
    parser.add_argument('--predictor',default='edge_stgru',choices=['edge_stgru','ridge'])
    parser.add_argument('--method',default='F4',choices=['F0','F1','F4'])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--budget',default='long',choices=['long','legacy'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage.startswith('parallel-'): orchestrate(args)
    else: worker(args)


if __name__ == '__main__': main()
