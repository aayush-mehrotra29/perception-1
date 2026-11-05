"""Real-cache audit of integer-second windows against original exact-query features."""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.fusion_data import load_grid
from brain.worker_contract import file_sha256
from .windows import pooled_windows
from .auxiliary import assemble

RENAMES={'text':'dense_text','audio':'dense_audio','video':'dense_video','tribe':'predicted_tribe'}


def audit(prepared,grids,output):
    prepared,grids,output=map(Path,(prepared,grids,output));guard(output)
    if output.exists():raise ValueError('preserve previous equivalence evidence')
    corpus=json.loads((prepared/'manifest.json').read_text());results=[]
    for lesson,record in corpus['feature_sources'].items():
        guard(output);path=Path(record['manifest'])
        if file_sha256(path)!=record['sha256']:raise ValueError('feature source changed')
        meta=json.loads(path.read_text());pins=meta['source_manifests']
        for pin in pins.values():
            if file_sha256(Path(pin['path']))!=pin['sha256']:raise ValueError('upstream source identity changed')
        grid_meta,blocks=load_grid(grids/lesson)
        if (grid_meta['source_sha256']!=meta['source_sha256'] or grid_meta['duration_seconds']!=meta['media_duration']
                or grid_meta['native_manifest_sha256']!=pins['tribe']['sha256']):raise ValueError('grid source/clock mismatch')
        feature_path=path.parent/'features.npz'
        if file_sha256(feature_path)!=meta['features_sha256']:raise ValueError('original features changed')
        with np.load(feature_path,allow_pickle=False) as z:
            queries=z['query_seconds'];expected={name:z[key] for name,key in RENAMES.items()}
            aux_expected={name:z[name] for name in ['semantics','sam_audio','sam_visual','progress','visual_descriptors','audio_descriptors']}
        selected=np.flatnonzero((queries>=44)&(queries==np.floor(queries)))
        lookup={int(queries[i]):int(i) for i in selected}
        if len(lookup)!=len(selected):raise ValueError('duplicate original integer query')
        compared={name:0 for name in blocks};max_error={name:0. for name in blocks};positions=0;new_supported=0
        for seconds,means,masks in pooled_windows(blocks,meta['media_duration'],batch_size=32):
            guard(output);positions+=len(seconds)
            for row,second in enumerate(seconds):
                if int(second) not in lookup:
                    if second>=44 and any(np.isfinite(v[row]).all() for name,v in means.items() if name!='tribe'):new_supported+=1
                    continue
                i=lookup[int(second)]
                for name,predicted in means.items():
                    actual=predicted[row];reference=expected[name][i]
                    if not np.array_equal(np.isfinite(actual),np.isfinite(reference)):raise ValueError('native support mask changed')
                    if np.isfinite(reference).all():
                        np.testing.assert_allclose(actual,reference,rtol=1e-5,atol=1e-6)
                        compared[name]+=1;max_error[name]=max(max_error[name],float(np.max(abs(actual-reference))))
        spec={'source_sha256':meta['source_sha256'],'media_duration':meta['media_duration'],'lecture':lesson,
            'sensory_data_root':str(Path(pins['native_sensory']['path']).parents[2])}
        for name in ['semantics','sam_visual','sam_audio','frames','audio_derivative']:spec[name]=str(Path(pins[name]['path']).parent)
        measured=assemble(spec,queries[selected],meta['source_sha256'],meta['media_duration'])
        aux_errors={}
        for name,values in measured.items():
            reference=aux_expected[name][selected]
            if not np.array_equal(np.isfinite(values),np.isfinite(reference)):raise ValueError('auxiliary support mask changed: '+name)
            valid=np.isfinite(reference)
            np.testing.assert_allclose(values[valid],reference[valid],rtol=1e-5,atol=1e-6)
            aux_errors[name]=float(np.max(abs(values[valid]-reference[valid]))) if valid.any() else None
        results.append({'lesson':lesson,'source_sha256':meta['source_sha256'],'grid_manifest_sha256':file_sha256(grids/lesson/'manifest.json'),
            'original_feature_manifest_sha256':record['sha256'],'one_second_positions':positions,'new_supported_integer_queries':new_supported,
            'native_compared_queries':compared,'native_max_abs_error':max_error,'auxiliary_max_abs_error':aux_errors,
            'passed':True,'native_TRIBE_source_written':False})
        del blocks,expected,aux_expected,measured
        print(lesson+': exact-query equivalence and new integer queries verified',flush=True)
    output.mkdir(parents=True)
    report={'schema':'perception-lesson-real-feature-equivalence-v1','complete':True,'lessons':results,
        'source_manifest_sha256':file_sha256(prepared/'manifest.json'),'code_sha256':file_sha256(Path(__file__)),
        'window_code_sha256':file_sha256(Path(__file__).with_name('windows.py')),'rtol':1e-5,'atol':1e-6,
        'prediction_interpolation_used':False,'one_second_psychological_accuracy_established':False,
        'purpose':'Engineering feature equivalence; not human/EEG/fMRI accuracy evidence'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['prepared','grids','output']:p.add_argument(name,type=Path)
    a=p.parse_args();audit(a.prepared,a.grids,a.output)
