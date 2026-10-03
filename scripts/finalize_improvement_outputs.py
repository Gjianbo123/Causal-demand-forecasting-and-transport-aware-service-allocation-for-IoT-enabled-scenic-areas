"""Wait for complete evidence, then audit, report, time, and compile in order."""
from pathlib import Path
import sys,subprocess,time,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/improvement_v4';OUT.mkdir(parents=True,exist_ok=True)
deadline=time.monotonic()+3600
while len(list((ROOT/'runs/improvement_v4/final_control').glob('*/*/day*/complete.json')))!=798:
    if time.monotonic()>deadline:raise TimeoutError('Control matrix remains incomplete; no manuscript generated')
    time.sleep(5)
print('Complete matrix found; starting independent audit',flush=True)
env=os.environ.copy();env['PYTHONUTF8']='1';env['TECTONIC_CACHE_DIR']=str(ROOT/'tmp/latex_tools/cache')
for name in ['audit_improvement_v4','analyze_improvement_v4','audit_improvement_summary','time_improvement_v4','draw_improvement_architecture','plot_improvement_results','build_improvement_manuscript']:
    print('STAGE',name,flush=True)
    with (OUT/(name+'.log')).open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts'/f'{name}.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for document in ['main','supplement']:
    print('COMPILE',document,flush=True)
    with (OUT/(document+'_compile.log')).open('w',encoding='utf8') as log:
        subprocess.run([str(ROOT/'tmp/latex_tools/tectonic.exe'),'-X','compile','--bundle','http://127.0.0.1:62526/tlextras-2022.0r0.tar',
                        '--only-cached','--keep-logs','--keep-intermediates',str(OUT/'Discover_IoT_LaTeX'/f'{document}.tex')],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Compiled outputs ready for visual QA; packaging is a separate audited step',flush=True)
