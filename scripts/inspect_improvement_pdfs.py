"""Structural checks and contact sheets for manual visual review."""
from pathlib import Path
import json,re
import pdfplumber
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/improvement_v4';QA=ROOT/'tmp/improvement_v4_pdf_qa'
report={}
for name in ['main','supplement']:
    doc=pdfplumber.open(OUT/'Discover_IoT_LaTeX'/f'{name}.pdf');full='\n'.join(p.extract_text() or '' for p in doc.pages)
    assert '??' not in full and '\ufffd' not in full
    bounds=[];sizes=[];upright_sizes=[];rotated_sizes=[]
    for index,page in enumerate(doc.pages):
        for char in page.chars:
            x0,y0,x1,y1=char['x0'],char['top'],char['x1'],char['bottom'];sizes.append(char['size'])
            if char['text'].strip():
                if char['upright']:upright_sizes.append(char['size'])
                else:rotated_sizes.append(x1-x0)
            if x0<0 or y0<0 or x1>page.width+.5 or y1>page.height+.5:bounds.append([index+1,char['text'],[x0,y0,x1,y1]])
    assert not bounds,bounds
    report[name]=dict(pages=len(doc.pages),text_characters=len(full),unresolved_references=False,outside_page_spans=bounds,
                      minimum_raw_vertical_glyph_extent=min(sizes),minimum_upright_visible_font_size=min(upright_sizes),
                      minimum_rotated_visible_font_extent=min(rotated_sizes) if rotated_sizes else None,
                      font_note='pdfplumber size is glyph width for rotated axis labels; upright and rotated extents are reported separately.',
                      text_by_page=[len(p.extract_text() or '') for p in doc.pages])
    (QA/f'{name}_extracted.txt').write_text(full,encoding='utf8')
    images=sorted(QA.glob(name+'-*.png'));assert len(images)==len(doc.pages)
    for start in range(0,len(images),6):
        subset=images[start:start+6];sheet=Image.new('RGB',(1800,1740),'#DADDE2');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(subset):
            im=Image.open(p).convert('RGB');im.thumbnail((590,835));x=(j%3)*600+5;y=(j//3)*870+25
            sheet.paste(im,(x,y));draw.text((x,y-20),p.stem,fill='black')
        sheet.save(QA/f'{name}_contact_{start//6+1}.png')
    doc.close()
(OUT/'PDF_STRUCTURE_QA.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
