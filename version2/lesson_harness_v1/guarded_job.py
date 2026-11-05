"""Run an explicitly named local annotation job with continuous disk guards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from brain.edugage_targets import guard


def run(module, checkpoint, output):
    if module not in {'qwen_teacher_diagnostic', 'qwen_teacher_corpus'}:
        raise ValueError('unsupported annotation job')
    output=Path(output).resolve();guard(output)
    status=output/'job-status.json';log=output/'runtime.log'
    if log.exists():raise ValueError('preserve previous job log')
    started=time.time();failure=None
    with log.open('x') as stream:
        process=subprocess.Popen([sys.executable,'-B','-m','lesson_harness_v1.'+module,
            str(Path(checkpoint).resolve()),str(output)],stdout=stream,stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[1])
        try:
            while process.poll() is None:
                guard(output)
                time.sleep(2)
        except BaseException as error:
            failure=type(error).__name__+': '+str(error)
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        finally:
            status.write_text(json.dumps({'module':module,'exit_code':process.returncode,
                'elapsed_seconds':time.time()-started,'failure':failure,
                'complete':process.returncode==0 and failure is None},indent=2)+'\n')
        if failure or process.returncode:
            raise RuntimeError(f'Local job failed; see {log}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('checkpoint');p.add_argument('output')
    a=p.parse_args();run(a.module,a.checkpoint,a.output)
