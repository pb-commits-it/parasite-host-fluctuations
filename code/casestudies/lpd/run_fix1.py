"""Rerun the registered dependency chain after archiving old checkpoints.

Uses supplied local data and registered seeds/budgets. Never edits report.md.
The initial archive procedure is recorded in fix1_plan.md; reusing a checkpoint
from another extraction is forbidden. See results/fix1_run_log.json for commands.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import subprocess
import time
from study_io import HERE,ROOT,RESULTS,save,now

PHASES=[('extract_series.py',[]),('analysis.py',['metrics']),('analysis.py',['predict']),
        ('refine_primary.py',[]),('summarize.py',[]),('record_inventory.py',[]),
        ('weather_continued.py',[]),('weather_summary.py',[]),('make_figures.py',[]),
        ('audit_results.py',[])]

def run():
    log=[]
    for script,args in PHASES:
        name=Path(script).stem+('_'+args[0] if args else '')
        cmd=[sys.executable,'-I',str(HERE/script),*args]
        record=dict(phase=name,command=cmd,started_utc=now());start=time.monotonic()
        print(record['started_utc'],name,'START',flush=True)
        with (RESULTS/(name+'_fix1_output.txt')).open('w') as out:
            result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        record.update(finished_utc=now(),seconds=time.monotonic()-start,returncode=result.returncode)
        log.append(record);save(RESULTS/'fix1_run_log.json',log)
        print(record['finished_utc'],name,'END',result.returncode,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    from compile_fix1_figures import run as compile_figures
    compile_figures()

if __name__=='__main__':run()
