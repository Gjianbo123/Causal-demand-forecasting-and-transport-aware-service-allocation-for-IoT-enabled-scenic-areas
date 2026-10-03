"""Audit v2 using its preserved source set, without treating added v3 modules as edits."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from iotexp.supplement_common import verify_v2,V3
import audit_rebuilt_results as original


if __name__=='__main__':
    # Verify the actual checkout's thirteen original modules and every asset first.
    verify_v2()
    # The old auditor uses ROOT only to enumerate/hash its original source set;
    # RUN and OUT still point to this checkout's v2 evidence and report directory.
    original.ROOT=V3/'frozen_v2_source'
    original.main()
