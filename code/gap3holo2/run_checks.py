"""Reproduce the entire independent GAP 3 attempt; no collaborator imports.
Run from any directory: python3 -I /path/to/gap3holo2/run_checks.py
Requires Python 3, mpmath and numpy. Writes coefficient artifacts beside this
script; stdout is the new run's log. The original failed logs are not touched.
"""
from pathlib import Path
import subprocess,sys,time
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
commands=[
 ['checks.py','--mode','gate'],
 ['checks.py','--mode','series','--order','32','--precision','100'],
 ['checks.py','--mode','series','--order','32','--precision','160'],
 ['high_order.py'],['analytic_checks.py'],['additional_checks.py'],
 ['baseline_checks.py'],['diagnostics.py']]
start=time.time()
for command in commands:
 print('\nCOMMAND:',sys.executable,'-I',str(P/command[0]),*command[1:],flush=True)
 subprocess.run([sys.executable,'-I',str(P/command[0]),*command[1:]],check=True)
print('\nPASS complete reproduction; elapsed seconds',round(time.time()-start,2),flush=True)
