"""Check the updated PDF and make contact sheets for visual inspection."""
from pathlib import Path
import json,re
import pdfplumber
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_more_figures'
DEST=ROOT/'output/improvement_v4_web/Discover_IoT_LaTeX'
QA=ROOT/'tmp/pdfs/improvement_v4_more_figures'

with pdfplumber.open(DEST/'main.pdf') as doc:
    texts=[p.extract_text() or '' for p in doc.pages];full='\n'.join(texts);bounds=[]
    for i,p in enumerate(doc.pages):
        bounds.extend([i+1,c['text']] for c in p.chars if c['x0']<0 or c['top']<0 or c['x1']>p.width+.5 or c['bottom']>p.height+.5)
    log=(DEST/'main.log').read_text(encoding='utf8',errors='replace')
    problems=re.findall(r'^.*(?:Overfull|undefined|Missing character|^!).*$',log,re.M|re.I)
    assert not bounds and not problems and '??' not in full and '\ufffd' not in full
    aux=(DEST/'main.aux').read_text(encoding='utf8')
    figures={label:dict(number=int(n),page=int(p)) for label,n,p in re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{(\d+)\}\{(\d+)\}',aux)}
    assert len(figures)==8 and sorted(x['number'] for x in figures.values())==list(range(1,9))
    report=dict(pages=len(doc.pages),unresolved_references=False,out_of_bounds=bounds,compile_errors=problems,figures=figures,text_by_page=[len(t) for t in texts])
    (QA/'main_extracted.txt').write_text(full,encoding='utf8')
    images=[QA/f'main-{i:02d}.png' for i in range(1,len(doc.pages)+1)]
    for start in range(0,len(images),6):
        sheet=Image.new('RGB',(1800,1740),'#DADDE2');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(images[start:start+6]):
            im=Image.open(p).convert('RGB');im.thumbnail((590,835));x=(j%3)*600+5;y=(j//3)*870+25
            sheet.paste(im,(x,y));draw.text((x,y-20),p.stem,fill='black')
        sheet.save(QA/f'contact_{start//6+1}.png')
    (OUT/'PDF_STRUCTURE_QA.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
