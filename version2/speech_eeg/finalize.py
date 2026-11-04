"""Bounded local handoff after the fixed training process completes successfully."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from .common import read,write,guard


def run(training_pid):
    resource=read('configs/stage_a.json');start=time.monotonic();run=Path('runs/cohort-v1')
    while not (run/'manifest.json').exists():
        guard(resource)
        try:os.kill(training_pid,0)
        except ProcessLookupError:raise RuntimeError('Training process ended without a complete manifest; inspect its log')
        if time.monotonic()-start>1800:raise RuntimeError('Bounded training observation expired; inspect the same process')
        time.sleep(10)
    if not read(run/'manifest.json')['complete']:raise ValueError('Incomplete training manifest')
    phases=[
        ('reload_audit',['speech_eeg.cohort_replay','runs/cohort-v1','data/cohort-features','data/cohort-targets']),
        ('reserved_test',['speech_eeg.cohort_evaluate','runs/cohort-v1','data/cohort-features','data/cohort-targets','runs/cohort-v1-evaluation']),
        ('artifact',['speech_eeg.inference','export','runs/cohort-v1','data/cohort-features','runs/cohort-v1-evaluation','artifacts/cohort-v1']),
        ('example',['speech_eeg.inference','predict','artifacts/cohort-v1','data/cohort-features/segment11.npz','runs/tribe-hidden-batch/segment11','runs/cohort-v1-example']),
        ('report',['speech_eeg.report','runs/cohort-v1','runs/cohort-v1-evaluation','runs/cohort-v1-report']),
        ('handoff',['speech_eeg.handoff','runs/cohort-v1','runs/cohort-v1-evaluation','artifacts/cohort-v1','runs/cohort-v1-report'])]
    for name,args in phases:
        guard(resource);write('runs/finalizer-status.json',{'phase':name,'complete':False})
        print('Starting '+name,flush=True)
        subprocess.run([sys.executable,'-u','-m',*args],check=True,timeout=900)
    write('runs/finalizer-status.json',{'phase':'handoff_written_needs_final_inspection','complete':True})
    print('Local handoff written. Inspect benchmark and rendered figures before completion.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('training_pid',type=int);a=p.parse_args();run(a.training_pid)
