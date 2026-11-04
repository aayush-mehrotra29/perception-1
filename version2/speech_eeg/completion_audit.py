"""Independent numerical and artifact checks for the finished research handoff."""
import argparse
from pathlib import Path
import numpy as np
from .common import read,write,sha
from .inference import authenticate_bundle


def run(output):
    root=Path('runs/cohort-v1');evaluation=Path('runs/cohort-v1-evaluation');artifact=Path('artifacts/cohort-v1');example=Path('runs/cohort-v1-example');report=Path('runs/cohort-v1-report')
    lock=read(root/'forecast-lock.json');r=read(evaluation/'results.json');c=read(root/'config.json')
    replay=read(root/'reload-audit.json');receipt=read(evaluation/'test-access-receipt.json')
    assert replay['passed'] and replay['forecast_lock_sha256']==receipt['forecast_lock_sha256']==sha(root/'forecast-lock.json')
    assert sha(root/'reload-audit.json')==receipt['reload_audit_sha256']
    assert (root/'forecast-lock.json').stat().st_mtime<=(root/'reload-audit.json').stat().st_mtime<=(evaluation/'test-access-receipt.json').stat().st_mtime
    assert len(lock['models'])==9 and len(lock['ridges'])==12 and len(replay['seed_forecast_sha256'])==54
    with np.load(root/'transforms.npz',allow_pickle=False) as z:mean=z['target_mean'];scale=z['target_scale']
    independent={a:[] for a in c['full_arms']};coverage={}
    for seg in c['test_segments']:
        yy=[];mm=[]
        for person in c['test_people']:
            tr=next(x for x in r['target_files'] if x['participant']==person and x['segment']==seg)
            path=evaluation/tr['file'];assert sha(path)==tr['sha256']
            assert path.stat().st_mtime>=(evaluation/'test-access-receipt.json').stat().st_mtime
            with np.load(path,allow_pickle=False) as z:yy.append(z['target']);mm.append(z['supported'])
        observed=(np.mean(yy,axis=0)-mean)/scale;mask=mm[0]&mm[1]
        pr=lock['forecasts'][f'segment{seg:02d}'];assert sha(root/pr['file'])==pr['sha256']
        with np.load(root/pr['file'],allow_pickle=False) as z:
            ids=z['query_indices'];valid=mask[ids];coverage[str(seg)]={'supported_channels':int(np.any(valid,0).sum()),'channel_sample_pairs':int(valid.sum())}
            for arm in c['full_arms']:
                error=(observed[ids]-z[arm])**2
                channel_error=[float(error[valid[:,j],j].mean()) for j in range(40) if valid[:,j].any()]
                value=float(np.mean(channel_error));independent[arm].append(value)
                np.testing.assert_allclose(value,r['primary_group']['per_clip'][str(seg)][arm]['MSE'],rtol=1e-12,atol=1e-12)
    macro={a:float(np.mean(v)) for a,v in independent.items()}
    for a,v in macro.items():np.testing.assert_allclose(v,r['primary_group']['equal_clip_channel_metrics'][a]['MSE'],rtol=1e-12,atol=1e-12)
    reduction=100*(macro['tribe_tcn']-macro['full_tcn'])/macro['tribe_tcn']
    np.testing.assert_allclose(reduction,r['primary_group']['full_vs_TRIBE_relative_MSE_reduction_percent'],atol=1e-12)
    card=authenticate_bundle(artifact);ex=read(example/'prediction.json');native=read('runs/tribe-hidden-batch/segment11/manifest.json')
    assert ex['model_artifact_sha256']==sha(artifact/'manifest.json') and ex['source']==native['source']
    assert ex['fMRI']['segments']==native['segments'] and ex['fMRI']['neuro_offset_seconds']==native['neuro_offset_seconds']
    assert sha(example/'native-fmri.npz')==sha('runs/tribe-hidden-batch/segment11/fmri.npz')==native['array_sha256']
    with np.load(example/'eeg.npz',allow_pickle=False) as z:
        pred=z['predicted_eeg_volts'];support=z['supported'];q=z['query_seconds']
        assert pred.shape==(len(q),40) and np.isfinite(pred[support]).all() and np.isnan(pred[~support]).all()
        assert ex['EEG']['supported']==support.tolist() and ex['EEG']['query_seconds']==q.tolist()
        assert all(all(v is None for v in row) for row,ok in zip(ex['EEG']['predicted'],support) if not ok)
        assert not any(v is None for row,ok in zip(ex['EEG']['predicted'],support) if ok for v in row)
        with np.load(root/lock['forecasts']['segment11']['file'],allow_pickle=False) as f:
            ids=f['query_indices'];valid=support[ids]
            expected=f['full_tcn']*scale+mean
            np.testing.assert_array_equal(pred[ids[valid]],expected[valid])
    for name,digest in read(report/'manifest.json')['files'].items():assert sha(report/name)==digest
    assert read('runs/resume-preservation-audit.json')['passed']
    assert read('runs/native-capture-audit.json')['all_passed']
    assert read('runs/native-encoder-identity-audit.json')['passed']
    assert read('runs/training-scaling-audit.json')['passed']
    assert read('runs/feature-assembly-audit.json')['passed']
    assert read('handoff-manifest.json')['complete']
    write(output,{'complete':True,'independent_primary_metrics':macro,'primary_reduction_percent':reduction,'test_coverage':coverage,
        'nine_models_and_fifty_four_seed_forecasts_replayed':True,'twelve_label_budget_ridges_present':True,
        'reserved_target_access_followed_forecast_and_reload_locks':True,'example_matches_evaluated_weights_exactly':True,
        'native_cortical_file_and_clock_unchanged':True,'JSON_missing_values_null':True,
        'artifact_manifest_sha256':sha(artifact/'manifest.json'),'result_sha256':sha(evaluation/'results.json'),
        'report_manifest_sha256':sha(report/'manifest.json'),'model_card_sha256':sha('MODEL_CARD.txt'),
        'integration_sha256':sha('INTEGRATION.txt'),'code_sha256':sha(__file__),
        'limits':['Auxiliary one-story EEG transfer test only','No measured-fMRI superiority','No educational outcomes validated','Arbitrary raw-upload parity not qualified']})
    print('Independent metrics, locked access order, model export and composite example all verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
