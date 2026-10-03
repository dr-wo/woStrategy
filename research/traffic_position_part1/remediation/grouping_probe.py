"""Compare the reported grouping shape against prerequisite, author and repair."""
import json
from pathlib import Path
import subprocess
import sys
import types
import pandas as pd
root=Path(__file__).resolve().parents[3]
commits=json.loads(Path(__file__).with_name('baseline_commits.json').read_text())
path='src/wostrategy/analysis/traffic.py'
frame=pd.DataFrame([dict(Year=2026,Driver=d,LapNumber=lap,SessionTime=start+i*.25,Speed=180.)
    for d,lap,start in [('A',10,0.),('B',9,.5)] for i in range(20)])
results={}
for label,ref in [('prerequisite','57495559a860814a96ade41f30dc74df699c9ed6'),
                  ('author',commits['woStrategy']),('remediated',None)]:
    text=(root/path).read_text() if ref is None else subprocess.check_output(['git','show',ref+':'+path],cwd=root,text=True)
    module=types.ModuleType('_grouping_'+label);sys.modules[module.__name__]=module
    exec(compile(text,label,'exec'),module.__dict__)
    try:
        result=module.evaluate_direct_fastf1_style(frame,group_columns=('Driver','LapNumber'))
        results[label]={'error':None,'lap_rows':len(result.laps),'keys':result.laps[['Driver','LapNumber']].to_dict('records')}
    except Exception as exc:results[label]={'error':str(type(exc).__name__)+': '+str(exc)}
assert results['prerequisite']['error'] is None
assert results['author']['error'].startswith('KeyError')
assert results['prerequisite']['keys']==results['remediated']['keys']
print(json.dumps(results,indent=2))
