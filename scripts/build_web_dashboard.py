"""Build a self-contained read-only UI from a fixed archived synthetic episode."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/improvement_v4_web'
EPISODE=ROOT/'runs/improvement_v4/final_control/20261011_nominal/rollout/day00'
DATA=ROOT/'runs/improvement_v4/final_datasets/20261011_nominal.npz'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);web=OUT/'web_dashboard';web.mkdir(exist_ok=True)
    record=json.loads((EPISODE/'complete.json').read_text())
    cfg=json.loads((ROOT/'runs/scenic_rebuild_v2/config.json').read_text())['scenic']
    with np.load(DATA) as z:
        edges=z['edge_index'].astype(int).tolist();dist=z['shortest_distance_m']
    speed=np.array(cfg['resource_speed_m_per_min']);setup=np.array(cfg['resource_setup_minutes'])
    travel=np.ceil((dist[None,:,:]/speed[:,None,None]+setup[:,None,None])/10).astype(int)
    frames=[]
    with np.load(EPISODE/'trace.npz') as z:
        for t in range(72):
            ledger=z['ledger_before'][t];inv=ledger[:96].reshape(4,24);transit=ledger[96:192].reshape(4,24)
            obs=z['observed'][t];q=obs[-1,:,1:]
            moves=[dict(resource=int(k+1),origin=int(i),destination=int(j),units=int(z['transfers'][t,k,i,j]),
                        transit_minutes=int(travel[k,i,j]*10)) for k,i,j in zip(*np.nonzero(z['transfers'][t]))]
            frames.append(dict(t=t,inflow=int(obs[-1,:,0].sum()),queued=int(q.sum()),queue_by_node=q.sum(1).astype(int).tolist(),
                queue_by_class=q.sum(0).astype(int).tolist(),stationary=int(inv.sum()),in_transit=int(transit.sum()),
                stationary_by_class=inv.sum(1).astype(int).tolist(),transit_by_class=transit.sum(1).astype(int).tolist(),
                history=obs[:,:,0].T.astype(float).tolist(),forecast=z['forecast'][t].astype(float).tolist(),moves=moves,
                event_time=int(z['source_event'][t,-1].max()),receipt_time=int(z['source_receipt'][t,-1].max()),
                feasible=not bool(z['execution_infeasible'][t])))
    assert frames[36]['inflow']==302 and frames[36]['queued']==323
    assert all(f['stationary']+f['in_transit']==82 for f in frames)
    payload=dict(world=20261011,physical_day=record['physical_day'],condition='Nominal',controller='Transport rollout',
                 snapshot_t=36,default_node=0,edges=edges,frames=frames)
    raw=json.dumps(payload,separators=(',',':'),ensure_ascii=False)
    (web/'replay_data.json').write_text(raw,encoding='utf8')
    html=(ROOT/'scripts/web_dashboard_template.html').read_text(encoding='utf8').replace('__REPLAY_DATA__',raw)
    (web/'index.html').write_text(html,encoding='utf8')
    manifest=dict(purpose='Offline browser prototype, a read-only visualization of an existing synthetic episode; no online inference or field connection.',
        world=20261011,physical_day=153,screenshot_decision_t=36,screenshot_node='S1',
        selection_rule='First listed nominal test world and first test day, midpoint decision t=36; not chosen by achieved performance.',
        sources={p.relative_to(ROOT).as_posix():digest(p) for p in [DATA,EPISODE/'trace.npz',EPISODE/'complete.json',ROOT/'runs/scenic_rebuild_v2/config.json']},
        replay_data_sha256=digest(web/'replay_data.json'),interface_sha256=digest(web/'index.html'),
        field_mapping={'inflow':'sum(observed[t,-1,:,0])','queue':'sum(observed[t,-1,:,1:]) before dispatch',
                       'stationary':'sum(ledger_before[t,:96])','in_transit':'sum(ledger_before[t,96:192])',
                       'graph':'edge_index from archived synthetic dataset','forecast':'forecast[t,node,:], leads [1,2,4,6]',
                       'transfers':'nonzero transfers[t,k,i,j]','travel':'ceil((shortest_distance/speed+setup)/10)*10 minutes'},
        disclosure='HTML/CSS/SVG and extraction code authored with OpenAI Codex assistance. Pixels are rendered by Chromium from this inspectable source and recorded arrays; no image-generation model is used.')
    (web/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (web/'README_ZH.txt').write_text('双击 index.html 即可离线打开。时点滑块与节点选择会更新已保存的仿真轨迹；导出按钮下载当前状态。\n论文截图固定为 world 20261011、原始第153日、决策 t=36、节点 S1。展示数值来自原有记录，不增加实验结果。\n这是只读回放界面原型，没有真实传感器连接、在线模型推理或现场执行控制。可检查 replay_data.json 和 provenance.json。\n',encoding='utf8')
    print(json.dumps(dict(output=str(web),frames=len(frames),snapshot={k:frames[36][k] for k in ['inflow','queued','stationary','in_transit']})))

if __name__=='__main__':main()
