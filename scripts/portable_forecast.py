"""Audit/report an extracted Windows reference archive without changing frozen files.

Only recorded absolute provenance paths are mapped to this checkout. Every source
and dataset hash is still checked; no model, prediction or metric is changed.
"""
from pathlib import Path, PureWindowsPath
import argparse
import copy
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import iotexp.forecast_supplement as forecast


def remap(recorded,old_root,current_root):
    relative=PureWindowsPath(recorded).relative_to(PureWindowsPath(old_root))
    if '..' in relative.parts:raise ValueError('Provenance path escapes the archived root')
    return current_root.joinpath(*relative.parts)


def relocated_verify(output):
    output=Path(output).resolve()
    path=output/'protocol.json'
    if forecast.sha256(path)!=(output/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol changed')
    stored=json.loads(path.read_text(encoding='utf-8'))
    mapped=copy.deepcopy(stored)
    dataset=remap(stored['source_dataset'],stored['root'],ROOT)
    if forecast.sha256(dataset)!=stored['source_dataset_sha256']:
        raise ValueError('Relocated original dataset hash mismatch')
    paths={}
    for recorded,expected in stored['source_hashes'].items():
        local=remap(recorded,stored['root'],ROOT)
        if forecast.sha256(local)!=expected:
            raise ValueError('Relocated frozen source hash mismatch: '+str(local))
        paths[str(local)]=expected
    mapped.update(root=str(ROOT),output=str(output),source_dataset=str(dataset),source_hashes=paths)
    return mapped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','report'])
    parser.add_argument('--output',default='runs/supplement_v3/forecast')
    args=parser.parse_args()
    # Patch the provenance resolver in memory only, before downstream imports.
    forecast.verify_protocol=relocated_verify
    if args.stage=='audit':
        from audit_forecast_supplement import audit
        audit(args.output)
    else:
        forecast.report_experiment(args.output)


if __name__=='__main__':main()
