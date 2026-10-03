"""Create a fresh extension-replay workspace, preserving the original evidence.

Copies the frozen v2 study and disclosed v3 plan into a NEW directory. It never
deletes, moves, edits or overwrites the existing experiment. Forecast path-specific
protocols must be newly frozen in the destination before any fitting/evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def copy_tree(source,target):
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    destination=args.destination.resolve()
    if destination.exists():raise FileExistsError('Choose a new, nonexistent directory: '+str(destination))
    if destination==ROOT or ROOT in destination.parents:
        raise ValueError('Place the replay beside or outside the source checkout, never inside it')
    destination.mkdir(parents=True,exist_ok=False)
    for folder in ['iotexp','scripts','tests','configs','runs/scenic_rebuild_v2','output/revision/Discover_IoT_LaTeX']:
        copy_tree(ROOT/folder,destination/folder)
    copy_file(ROOT/'requirements-rebuilt.txt',destination/'requirements-rebuilt.txt')
    base=Path('runs/supplement_v3')
    for name in ['protocol.json','protocol.sha256','protocol_clarification_01.json','protocol_clarification_01.sha256']:
        copy_file(ROOT/base/name,destination/base/name)
    copy_tree(ROOT/base/'frozen_v2_source',destination/base/'frozen_v2_source')
    copy_tree(ROOT/base/'forecast/source',destination/base/'forecast/source')
    copy_file(ROOT/base/'forecast/THIRD_PARTY_NOTICES.md',destination/base/'forecast/THIRD_PARTY_NOTICES.md')
    # This is prior code-test evidence, not a final-data result or a test rerun.
    test_record_relative=base/'forecast/unit_test_results.json'
    test_record_source=ROOT/test_record_relative
    copy_file(test_record_source,destination/test_record_relative)
    test_record=json.loads(test_record_source.read_text(encoding='utf-8'))
    copy_file(ROOT/base/'control/solver_protocol.json',destination/base/'control/solver_protocol.json')
    # Reporting sources contain no final extended outcomes; they are templates.
    for name in ['framing_additions.tex','method_additions.tex','supplement_methods.tex','README_REPRODUCE_ZH.txt','journal_guidelines_check.txt']:
        copy_file(ROOT/'output/supplement_v3'/name,destination/'output/supplement_v3'/name)
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),origin=str(ROOT),destination=str(destination),
                purpose='Fresh replay of the extended experiment with preserved original v2 inputs',
                original_modified=False,original_v3_results_copied=False,
                preserved_unit_test_evidence=dict(file=test_record_relative.as_posix(),
                    record_sha256=hashlib.sha256(test_record_source.read_bytes()).hexdigest(),
                    original_recorded_utc=test_record['recorded_utc'],
                    original_mtime_ns=test_record_source.stat().st_mtime_ns,
                    tested_source_sha256=test_record['test_file_sha256'],
                    copied_with_original_mtime=True,reexecuted_in_replay_environment=False,
                    scope='Preserved original code-unit-test record; does not assert tests ran on the new machine. '
                          'Run python -m unittest discover -s tests -v in the new workspace.'),
                next_step='Freeze forecast and matched-ablation protocols locally before training. Follow README_REPRODUCE_ZH.txt.',
                reproducibility_limits='Reference Windows/CUDA environment; solver time limits and hardware kernels can change exact refitted trajectories. Stored-output numeric audit is distinct from fresh fitting.')
    (destination/'REPLAY_ORIGIN.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
