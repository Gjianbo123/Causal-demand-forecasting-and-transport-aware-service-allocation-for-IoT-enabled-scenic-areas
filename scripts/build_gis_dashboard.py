"""Build a dark GIS command-centre prototype with separated replay and demo data."""
from pathlib import Path
import json, hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/improvement_v4_gis'
WEB = OUT / 'web_dashboard'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'output/improvement_v4_web/web_dashboard'
    raw = (source / 'replay_data.json').read_text(encoding='utf8')
    original = json.loads((source / 'provenance.json').read_text(encoding='utf8'))
    assert sha(source / 'replay_data.json') == original['replay_data_sha256']
    demo = dict(purpose='Illustrative UI scenario only; excluded from experiments and not live observations.',
                reservations=28465, tickets=24312, revenue=1286520,
                ticket_channels=[42.6, 28.2, 14.7, 8.9, 5.6],
                weather=[dict(day='Today', temperature='18–26', condition='cloud'), dict(day='Tomorrow', temperature='19–27', condition='sun'), dict(day='Day 3', temperature='18–24', condition='rain')],
                parking=[dict(id='P1',capacity=420,available=86),dict(id='P2',capacity=360,available=124),dict(id='P3',capacity=280,available=42),dict(id='P4',capacity=300,available=168)],
                incidents=[dict(id='EV-001',kind='assistance',node=8,status=0,time='14:06'),dict(id='EV-002',kind='equipment',node=17,status=1,time='13:58'),dict(id='EV-003',kind='lost',node=4,status=2,time='13:42')],
                activities=[dict(time='09:00',key='heritage',status=2),dict(time='10:00',key='nature',status=2),dict(time='14:00',key='forest',status=1),dict(time='16:00',key='valley',status=0),dict(time='19:30',key='music',status=0)])
    (WEB / 'replay_data.json').write_text(raw, encoding='utf8')
    (WEB / 'illustrative_scenario.json').write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding='utf8')
    template = (ROOT / 'scripts/scenic_command_template.html').read_text(encoding='utf8')
    template = template.replace('__REPLAY_DATA__', raw).replace('__DEMO_DATA__', json.dumps(demo, ensure_ascii=False))
    template = template.replace('__MAP_SCRIPT__', (ROOT / 'scripts/scenic_gis_map.js').read_text(encoding='utf8'))
    assert '__REPLAY_DATA__' not in template and '__MAP_SCRIPT__' not in template
    (WEB / 'index.html').write_text(template, encoding='utf8')
    original.update(purpose='Bilingual, offline GIS-style command-centre prototype. Recorded synthetic experiment panels and illustrative business panels are explicitly separated.',
                    interface_sha256=sha(WEB / 'index.html'),
                    illustrative_scenario_sha256=sha(WEB / 'illustrative_scenario.json'),
                    map='Original procedural SVG schematic, not surveyed GIS coordinates or a real scenic site.',
                    reference='User-provided smart-tourism command-centre screenshot inspired the dark, map-centred information hierarchy; no reference image pixels or branding are embedded.',
                    demonstration_fields=['ticketing', 'parking', 'weather', 'incidents', 'activities'],
                    browser_features=['Chinese/English switch', 'six navigation views', 'time replay', 'node selection', 'map layers', 'zoom', 'incident scenario steps', 'state export'])
    (WEB / 'provenance.json').write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding='utf8')
    (WEB / 'README_ZH.txt').write_text('''智慧景区文旅运营指挥平台 / 离线网页演示
双击 index.html 即可打开；无需安装、联网或启动服务器。右上角可切换中英文和全屏；底部控制回放时点；地图可选节点、缩放和切换图层。
六个导航页对应综合态势、客流管控、票务经营、交通停车、应急指挥、游客服务。事件条目可打开演示处置时间线，操作仅修改浏览器内的示例状态。导出按钮将仿真记录和界面示例分开保存。
青色“仿真回放”模块来自既有实验轨迹：流入、队列、预测、资源与转移。金色“界面示例”模块为设计场景：票务、停车、天气、事件与活动，不能用于实验结论。
地图由原创SVG代码绘制，为景区空间示意，非真实GIS坐标。没有连接票务、摄像头、停车场、传感器或应急通知系统。
论文使用英文界面截图；中文版本用于演示。可在文件URL末尾加 ?lang=en 打开英文版。源数据和字段映射见 replay_data.json、illustrative_scenario.json、provenance.json。
''', encoding='utf8')
    print(json.dumps(dict(output=str(WEB), replay_sha256=sha(WEB / 'replay_data.json'), interface_sha256=sha(WEB / 'index.html'))))


if __name__ == '__main__':
    main()
