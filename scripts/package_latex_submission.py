"""Package LaTeX submission sources after successful compilation and visual QA."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'output/revision'
SRC=REV/'Discover_IoT_LaTeX'


def main():
    pdf=SRC/'main.pdf'
    assert pdf.exists() and len(PdfReader(pdf).pages)>0
    assert (SRC/'main.bbl').exists()
    log=(SRC/'main.log').read_text(encoding='utf-8',errors='replace')
    for failure in ('There were undefined references','There were undefined citations','! Emergency stop','! Undefined control sequence'):
        assert failure not in log,failure
    files=[SRC/name for name in ('main.tex','references.bib','main.bbl','sn-jnl.cls','sn-mathphys-num.bst','cover_letter.tex','README_ZH.md','conversion_audit.json')]
    for optional in ('LATEX_CONTENT_AUDIT.md','LATEX_CONTENT_AUDIT.json','COMPILE_AND_LAYOUT_QA.md'):
        if (SRC/optional).exists(): files.append(SRC/optional)
    files+=sorted((SRC/'figures').glob('*.pdf'))
    assert len(list((SRC/'figures').glob('*.pdf')))==5
    manifest=[]
    target=REV/'Discover_IoT_LaTeX.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            data=path.read_bytes(); name=path.relative_to(SRC).as_posix()
            z.writestr(name,data)
            manifest.append(hashlib.sha256(data).hexdigest()+'  '+name)
        z.writestr('MANIFEST.sha256','\n'.join(manifest)+'\n')
    with zipfile.ZipFile(target) as z: assert z.testzip() is None
    preview=REV/'Discover_IoT_LaTeX_Preview.pdf'
    shutil.copy2(pdf,preview)
    print(json.dumps({'source_archive':str(target),'files':len(files)+1,'archive_bytes':target.stat().st_size,
                      'compiled_pdf':str(preview),'pages':len(PdfReader(preview).pages)}))


if __name__=='__main__': main()
