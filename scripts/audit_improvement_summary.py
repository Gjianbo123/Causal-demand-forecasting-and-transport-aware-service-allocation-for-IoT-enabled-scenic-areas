"""Independently recompute published summaries from archived episode records."""
from pathlib import Path
import sys,json,csv,math
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from iotexp.util import write_json,digest
V4=ROOT/'runs/improvement_v4';OUT=ROOT/'output/improvement_v4'

def main():
    records=[json.loads(p.read_text()) for p in (V4/'final_control').glob('*/*/day*/complete.json')]
    worlds=defaultdict(list)
    for record in records:
        rs=record['results'];n=len(rs)
        assert n==(10 if record['method'].startswith('ppo') else 1)
        if n==10:assert sorted(r['training_seed'] for r in rs)==list(range(10))
        assert len({r['arrivals_count'] for r in rs})==1
        values=dict(area=sum(r['waiting_area_min'] for r in rs)/n,arrivals=rs[0]['arrivals_count'],
                    left=sum(r['unserved_count'] for r in rs)/n,movement=sum(r['movement_cost'] for r in rs)/n,
                    overload=sum(r['visitor_overload_count'] for r in rs)/n,den=sum(r['visitor_overload_denominator'] for r in rs)/n,
                    reward=sum(r['mean_reward'] for r in rs)/n,day=record['day'])
        worlds[(record['dataset_seed'],record['surge'],record['method'])].append(values)
    grouped=defaultdict(list);cells=0
    for row in csv.DictReader((OUT/'control_world_metrics.csv').open(encoding='utf8')):
        key=(int(row['dataset_seed']),row['surge']=='True',row['method']);days=worlds[key]
        if row['subset']=='matched12':days=[d for d in days if d['day'] in [0,12,24]]
        assert len(days)==(3 if row['subset']=='matched12' else 9)
        a=sum(d['arrivals'] for d in days);total=lambda name:sum(d[name] for d in days)
        expected=dict(wait=total('area')/a,unserved_pct=100*total('left')/a,
                      overload_pct=100*total('overload')/total('den'),movement=total('movement')/len(days),reward=total('reward')/len(days))
        for name,value in expected.items():
            assert math.isclose(float(row[name]),value,rel_tol=1e-10,abs_tol=1e-10),(key,name)
            cells+=1
        grouped[(row['surge'],row['subset'],row['method'])].append(expected)
    for row in csv.DictReader((OUT/'control_summary.csv').open(encoding='utf8')):
        values=grouped[(row['surge'],row['subset'],row['method'])];assert len(values)==7
        for key in ['wait','unserved_pct','overload_pct','movement','reward']:
            x=np.array([v[key] for v in values]);mean=x.sum()/7;sd=np.sqrt(((x-mean)**2).sum()/6)
            assert math.isclose(float(row[key+'_mean']),mean,rel_tol=1e-10,abs_tol=1e-10)
            assert math.isclose(float(row[key+'_std']),sd,rel_tol=1e-10,abs_tol=1e-10);cells+=2
    fg=defaultdict(list)
    for path in (V4/'final_forecast').glob('*/complete.json'):
        for r in json.loads(path.read_text())['scores']:
            fg[(str(r['surge']),r['model'])].append(r)
    for row in csv.DictReader((OUT/'forecast_summary.csv').open(encoding='utf8')):
        rr=fg[(row['surge'],row['model'])];assert len(rr)==7
        for key in ['mae','rmse','wape','mae_h1','mae_h2','mae_h4','mae_h6']:
            x=np.array([r[key] if key in r else r['horizon_mae'][[1,2,4,6].index(int(key[5:]))] for r in rr])
            assert np.isclose(float(row[key+'_mean']),x.mean(),atol=1e-10,rtol=1e-10)
            assert np.isclose(float(row[key+'_std']),x.std(ddof=1),atol=1e-10,rtol=1e-10);cells+=2
    result=json.loads((OUT/'results.json').read_text())
    assert result['episodes']==3066 and result['decision_epochs']==220752
    write_json(OUT/'REPORT_AUDIT.json',dict(status='PASS',independently_checked_world_and_summary_cells=cells,
        complete_control_cases=len(records),results_sha256=digest(OUT/'results.json'),
        checks=['correct per-world request pooling','PPO training seeds averaged before world comparison','same horizon-comparison day subset','all seven worlds per summary','sample rather than population SD','source forecast metrics and per-lead mapping']))
    print('REPORT AUDIT PASS',cells,flush=True)

if __name__=='__main__':main()
