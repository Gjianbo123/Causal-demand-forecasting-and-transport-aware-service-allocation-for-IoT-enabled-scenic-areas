from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/improvement_v4/final_control';out.mkdir(parents=True,exist_ok=True)
children=[]
for wid in range(4):
    log=(out/f'worker{wid}.log').open('a',encoding='utf-8')
    command=[sys.executable,str(ROOT/'scripts/run_improvement_final.py'),'worker','--worker-id',str(wid),'--workers','4']
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    children.append((child,log));print('Started worker',wid,'pid',child.pid,flush=True)
statuses=[]
for child,log in children:statuses.append(child.wait());log.close()
print('Workers complete:',statuses,flush=True)
if any(statuses):raise RuntimeError('Worker failed; inspect preserved log and traces')
