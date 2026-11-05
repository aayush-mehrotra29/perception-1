"""Audited Qwen corpus → fixed six-arm benchmark → separate deployment fit."""
import argparse
import json
from pathlib import Path
import time

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .qwen_annotation_import import prepare
from .qwen_benchmark import fit as fit_benchmark, score
from .qwen_deployment import fit as fit_deployment


def run(source, packets, teacher, output):
    output=Path(output);guard(output)
    if output.exists():raise ValueError('choose a new release directory; preserve previous attempts')
    output.mkdir(parents=True)
    started=time.monotonic()
    progress={'complete':False,'phase':'import','source':str(Path(source).resolve()),
              'teacher':str(Path(teacher).resolve()),'participant_data_sent_to_cloud':False}
    def save(phase):
        progress.update(phase=phase,elapsed_seconds=time.monotonic()-started)
        (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(phase,flush=True)
    try:
        save('authenticating_and_importing_labels')
        imported=prepare(source,packets,teacher,output/'corpus')
        progress['quality_origin_counts']=imported['quality_origin_counts']
        save('fitting_all_prespecified_arms')
        fit_benchmark(output/'corpus',output/'benchmark')
        save('scoring_saved_heldout_predictions')
        score(output/'corpus',output/'benchmark')
        save('fitting_separate_full_data_artifact')
        fit_deployment(output/'corpus',output/'benchmark',output/'deployment')
        progress.update(complete=True,deployment=str((output/'deployment').resolve()),
            deployment_manifest_sha256=file_sha256(output/'deployment/manifest.json'),
            benchmark_sha256=file_sha256(output/'benchmark/benchmark.json'))
        save('complete')
    except BaseException as error:
        progress['error']=type(error).__name__+': '+str(error);save('stopped')
        raise
    return progress


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','packets','teacher','output'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.source,a.packets,a.teacher,a.output)
