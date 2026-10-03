"""Package complete audited evidence, excluding unrelated user files."""
from pathlib import Path
import json,sys,zipfile,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from iotexp.util import digest,write_json
V4=ROOT/'runs/improvement_v4';OUT=ROOT/'output/improvement_v4'

def main():
    audit=json.loads((V4/'INDEPENDENT_AUDIT.json').read_text());assert audit['status']=='PASS'
    qa=json.loads((OUT/'FINAL_QA.json').read_text());assert qa['status']=='PASS'
    target=OUT/'Discover_IoT_Improved_LaTeX.zip'
    source=OUT/'Discover_IoT_LaTeX'
    manuscript=[p for p in source.rglob('*') if p.is_file() and p.suffix in ['.tex','.bib','.bbl','.cls','.bst','.pdf','.svg','.png','.eps','.txt']]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(manuscript):z.write(p,p.relative_to(source).as_posix())
    evidence=[]
    for folder in ['iotexp','scripts','tests','runs/scenic_rebuild_v2/assets','runs/supplement_v3/frozen_v2_source','runs/supplement_v3/forecast',
                   'runs/supplement_v3/policies/edge_stgru/checkpoints','runs/improvement_v4','output/improvement_v4/Discover_IoT_LaTeX',
                   'output/improvement_v4/figures','output/supplement_v3/Discover_IoT_LaTeX']:
        evidence += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.log','.aux','.out','.blg']]
    evidence += [ROOT/p for p in ['requirements-rebuilt.txt','runs/scenic_rebuild_v2/config.json','runs/scenic_rebuild_v2/manifest.json']]
    evidence += [p for p in OUT.glob('*') if p.is_file() and p.suffix in ['.json','.txt','.csv'] and p.name!='package_manifest.json']
    evidence=sorted(set(evidence));files={p.relative_to(ROOT).as_posix():digest(p) for p in evidence}
    package=OUT/'ScenicIoT_Improved_Reproducibility.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in evidence:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('EVIDENCE_SHA256.json',json.dumps(files,ensure_ascii=False,indent=2))
    for p in [target,package]:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise RuntimeError('Corrupt archive')
    write_json(OUT/'package_manifest.json',dict(latex=dict(file=target.name,sha256=digest(target),bytes=target.stat().st_size,files=len(manuscript)),
        reproducibility=dict(file=package.name,sha256=digest(package),bytes=package.stat().st_size,files=len(evidence)+1)))
    print('Packaged',len(manuscript),len(evidence),'files',flush=True)

if __name__=='__main__':main()
