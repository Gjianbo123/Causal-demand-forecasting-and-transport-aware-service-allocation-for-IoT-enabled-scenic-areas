"""Package compiled LaTeX sources and the complete reproducibility evidence.

Run only after all experimental and visual audits have finished.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/supplement_v3'
SOURCE=OUT/'Discover_IoT_LaTeX'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def archive(target,pairs):
    names=set();manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as bundle:
        for path,name in pairs:
            assert name not in names,name
            names.add(name)
            bundle.write(path,name)
            manifest.append(sha(path)+'  '+name)
        bundle.writestr('MANIFEST.sha256','\n'.join(manifest)+'\n')
    with zipfile.ZipFile(target) as bundle:
        error=bundle.testzip()
        assert error is None,error
    target.with_suffix(target.suffix+'.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii')
    return dict(path=str(target),files=len(names)+1,bytes=target.stat().st_size,sha256=sha(target))


def main():
    qa=json.loads((OUT/'FINAL_QA.json').read_text())
    assert qa['status']=='PASS'
    for stem in ['main','supplement']:
        assert (SOURCE/(stem+'.pdf')).exists() and (SOURCE/(stem+'.bbl')).exists()
        log=(SOURCE/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
        for message in ['There were undefined references','There were undefined citations','! Emergency stop','! Undefined control sequence','Overfull \\hbox','Overfull \\vbox']:
            assert message not in log,(stem,message)
    source_names=['main.tex','supplement.tex','supplement_methods.tex','references.bib','main.bbl','supplement.bbl',
                  'sn-jnl.cls','sn-mathphys-num.bst','README.txt','cover_letter.tex']
    source_files=[(SOURCE/name,name) for name in source_names]
    source_files += [(p,p.relative_to(SOURCE).as_posix()) for p in sorted((SOURCE/'figures').glob('*.pdf'))]
    for extension in ['svg','eps']:
        source_files += [(p,'editable_figures/'+p.name) for p in sorted((OUT/'figures').glob('*.'+extension))]
    source_files += [(OUT/'FINAL_QA.json','FINAL_QA.json')]
    sources=archive(OUT/'Discover_IoT_Extended_LaTeX.zip',source_files)
    shutil.copy2(SOURCE/'main.pdf',OUT/'Discover_IoT_Extended_Manuscript.pdf')
    shutil.copy2(SOURCE/'supplement.pdf',OUT/'Discover_IoT_Extended_Supplement.pdf')

    files=[]
    for folder in ['iotexp','scripts','tests','configs','runs/scenic_rebuild_v2','runs/supplement_v3']:
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
            if '.partial.' in path.name or '.in_progress.' in path.name:continue
            files.append((path,path.relative_to(ROOT).as_posix()))
    files.append((ROOT/'requirements-rebuilt.txt','requirements-rebuilt.txt'))
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or 'qa_pages' in path.parts:continue
        if path.suffix in ['.zip','.sha256','.aux','.log','.blg','.out','.synctex','.gz']:continue
        if path.name=='package_manifest.json':continue
        files.append((path,path.relative_to(ROOT).as_posix()))
    # The manuscript builder retains the original Springer article as a source.
    for path in sorted((ROOT/'output/revision/Discover_IoT_LaTeX').rglob('*')):
        if path.is_file() and path.suffix in ['.tex','.bib','.bst','.cls','.pdf']:
            files.append((path,path.relative_to(ROOT).as_posix()))
    for name in ['manuscript_results.json','forecast_results.csv','contrasts_with_intervals.csv','final_numeric_audit.json','README_REPRODUCE_ZH.md']:
        path=ROOT/'output/revision'/name
        if path.exists():files.append((path,path.relative_to(ROOT).as_posix()))
    files.append((OUT/'README_REPRODUCE_ZH.txt','README_REPRODUCE_ZH.txt'))
    evidence=archive(OUT/'ScenicIoT_Extended_Reproducibility.zip',files)
    report=dict(latex_sources=sources,reproducibility=evidence,qa=qa,
                compiled_main=str(OUT/'Discover_IoT_Extended_Manuscript.pdf'),
                compiled_supplement=str(OUT/'Discover_IoT_Extended_Supplement.pdf'))
    (OUT/'package_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
