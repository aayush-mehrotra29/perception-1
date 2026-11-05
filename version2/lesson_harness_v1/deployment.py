"""Separate full-data fit after matched evaluation; no benchmark-driven arm choice."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES,VERSION
from .artifact import preflight
from .data import load_import
from .eeg_auxiliary import prepare as prepare_eeg
from .features import WIDTH,fit_transforms,save_transforms,transform
from .quality_import import validate_qualification
from .training import fit_network


def head_status(observations,benchmark):
    result={}
    for quality in QUALITIES:
        labels=[o.label for o in observations if o.label.quality==quality]
        if not labels:result[quality]={'status':'blocked','reason':'no_accepted_training_labels'};continue
        origins=sorted({l.origin for l in labels})
        human=benchmark['endpoints'][quality]['human']
        result[quality]={'status':'fitted','training_labels':len(labels),'supervision_origins':origins,
            'validation_status':'human_evaluated' if human['status']=='evaluated' else 'exploratory',
            'training_label_resolution_seconds':None,
            'label_scope':'sporadic original human intervals and/or locally generated 44-second synthetic interpretations; not one-second labels',
            'human_accuracy_improvement_established':False}
    return result


def fit(source,benchmark_directory,output):
    source,benchmark_directory,output=map(Path,(source,benchmark_directory,output));guard(output)
    if output.exists():raise ValueError('preserve previous deployment artifact')
    manifest,blocks,observations,windows,folds,eeg=load_import(source);validate_qualification(manifest,eeg)
    if manifest.get('schema')!='perception-lesson-labeled-import-v1':raise ValueError('audited full label corpus required')
    benchmark=json.loads((benchmark_directory/'benchmark.json').read_text())
    protocol=json.loads((benchmark_directory/'protocol.json').read_text())
    fit_report=json.loads((benchmark_directory/'manifest.json').read_text())
    config_path=Path(__file__).with_name('config.json');config=json.loads(config_path.read_text())
    if (not benchmark['complete'] or benchmark.get('schema')!='perception-lesson-five-head-benchmark-v1'
            or not fit_report['complete'] or benchmark['fit_manifest_sha256']!=file_sha256(benchmark_directory/'manifest.json')
            or fit_report['protocol_sha256']!=file_sha256(benchmark_directory/'protocol.json')
            or protocol['source_manifest_sha256']!=file_sha256(source/'manifest.json')
            or benchmark['source_manifest_sha256']!=protocol['source_manifest_sha256']
            or protocol['config_sha256']!=file_sha256(config_path)):
        raise ValueError('complete matching benchmark and fixed configuration required')
    for name in ['model.py','training.py','features.py','eeg_auxiliary.py']:
        if protocol['code'][name]!=file_sha256(Path(__file__).with_name(name)):
            raise ValueError('deployment training code differs from evaluated code')
    output.mkdir(parents=True)
    indices=sorted({w for o in observations for w in o.window_indices}|set(eeg['point'].tolist()))
    states=fit_transforms(blocks,np.asarray(indices));save_transforms(states,output/'transforms',indices)
    x=transform(blocks,states);aux=prepare_eeg(eeg,np.arange(len(eeg['point'])))
    members=[];seed_training=[]
    for seed in config['training']['seeds']:
        run=output/'ensemble'/str(seed)
        trained=fit_network(x,observations,windows,run,seed=seed,arm='full_eeg_informed',eeg=aux)
        if not trained['reload_verification']['exact']:raise ValueError('reload verification required')
        member_path=run/'network/manifest.json'
        members.append({'seed':seed,'manifest':str(member_path.relative_to(output)),'sha256':file_sha256(member_path)})
        seed_training.append({'seed':seed,'manifest':str((run/'training.json').relative_to(output)),
            'sha256':file_sha256(run/'training.json'),'reload_verification':trained['reload_verification']})
    training={'schema':'perception-lesson-deployment-training-v1','complete':True,'purpose':'full_data_deployment',
        'arm':'full_eeg_informed','arm_selected_from_benchmark':False,'source_manifest_sha256':file_sha256(source/'manifest.json'),
        'source':str(source.resolve()),'seeds':seed_training,'unique_training_windows':len(indices),
        'training_predictions_used_as_validation':False,'all_participant_targets_local':True}
    (output/'training.json').write_text(json.dumps(training,indent=2)+'\n')
    for origin,name in [(config_path,'config.json'),(benchmark_directory/'benchmark.json','benchmark.json'),
                        (source/'label-audit.json','label-audit.json'),(benchmark_directory/'protocol.json','evaluation-protocol.json')]:
        shutil.copyfile(origin,output/name)
    status=head_status(observations,benchmark)
    report={'schema':'perception-lesson-deployment-v1','model_version':VERSION,'deployment_status':'fitted',
        'synthetic_fixture':False,'input_width':WIDTH,'qualities':list(QUALITIES),'context_seconds':44,'query_step_seconds':1,
        'reference_population':config['reference_population'],'heads':status,'ensemble':members,
        'files':{name:file_sha256(output/name) for name in ['config.json','training.json','benchmark.json','label-audit.json','evaluation-protocol.json','transforms/manifest.json']},
        'EEG_role':'training-only screened received-voltage auxiliary; recorded EEG is not an inference input',
        'TRIBE_role':'unchanged frozen native cortical output, separately preserved',
        'one_second_psychological_accuracy_established':False,'individual_student_calibration_established':False,
        'measured_fmri_superiority_established':False,'document_inputs':'experimental transfer using explicit presentation schedules'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    preflight(output)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','benchmark','output']:p.add_argument(name,type=Path)
    a=p.parse_args();r=fit(a.source,a.benchmark,a.output)
    print(json.dumps({'deployment_status':r['deployment_status'],'heads':r['heads']}),flush=True)
