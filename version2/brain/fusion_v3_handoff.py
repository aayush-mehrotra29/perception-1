"""Package the v3 revision with complete results and unchanged-cortex evidence."""
import argparse
from datetime import datetime,timezone
import html
from pathlib import Path
import shutil
from .fusion_data import read_json,write_json
from .fusion_inference import validate_artifact
from .worker_contract import file_sha256


def run(runs,output):
    runs,output=Path(runs),Path(output)
    report=read_json(runs/'fusion-v3-benchmark/report.json')
    audit=read_json(runs/'fusion-v3-audit/audit.json')
    qualification=read_json(runs/'fusion-v3-qualification/qualification.json')
    model=runs/'fusion-v3-deployment';model_meta=validate_artifact(model)
    assert audit['status']==qualification['status']=='passed'
    assert qualification['model_manifest_sha256']==file_sha256(model/'manifest.json')
    assert audit['evaluation_manifest_sha256']==report['evaluation_manifest_sha256']
    assert file_sha256(runs/'fusion-v3-benchmark/report.json')==model_meta['benchmark_sha256']
    output.mkdir(parents=True,exist_ok=False)
    for source,dest in [(model,'model'),(runs/'fusion-v3-benchmark','benchmark'),
                        (runs/'fusion-v3-qualification/video/prediction','video-example')]:shutil.copytree(source,output/dest)
    for source,dest in [(runs/'fusion-v3-audit/audit.json','audit.json'),
                        (runs/'fusion-v3-qualification/qualification.json','qualification.json'),
                        (runs/'fusion-v3-quality/quality.json','signal-quality.json'),
                        (Path('configs/brain_fusion_v3.json'),'protocol.json')]:shutil.copy2(source,output/dest)
    for name in ['fusion-v3-tests-final-base.log','fusion-v3-tests-torch.log']:
        shutil.copy2(runs/name,output/name)
    previous=read_json(Path('docs/brain_model_card_v2.json'))
    metric={k:v['primary_normalized_MAE'] for k,v in report['metrics'].items()}
    primary=report['comparisons']['tribe_temporal'];passed=report['success_target_point_estimate_met']
    conclusion='Prespecified 5% point-estimate improvement target met.' if passed else 'Prespecified 5% improvement target not met.'
    if primary['relative_error_reduction']>0 and not report['primary_interval_excludes_zero_in_favorable_direction']:
        conclusion+=' Lower point error has uncertainty spanning no improvement.'
    control=('Shuffled correction matches or beats the correctly paired correction; no demonstrated benefit of temporal EEG pairing.'
             if metric['shuffled_correction']<=metric['fusion'] else
             'Correct pairing has lower point error than shuffled correction; inspect the descriptive interval before interpreting it.')
    card={'schema':'perception-brain-model-card-v3','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'architecture_runs':True,'EEG_benchmark_conclusion':conclusion,'native_fmri_modified':False,
        'measured_fmri_superiority_tested':False,'scope':'Competition-only exploratory revision; standalone research untouched',
        'architecture':{
            'frozen':previous['architecture']['frozen'],
            'anchor':'Three-seed TRIBE-only EEG temporal network; same source architecture as v2 comparator',
            'correction':'Training-only robust transforms and strongly regularized multimodal ridge residual, bounded by 0.1 target SD',
            'features':'Llama, W2v-BERT, V-JEPA2, TRIBE sequences plus existing SAM and content descriptors',
            'preprocessing':'Training 1st/99th percentiles, near-constant feature removal, bounded standardized values and PCA scores, auxiliary gain 0.25',
            'temporal_context':'44 seconds at 2 Hz; ordered-bin mean, standard deviation and half-context change for the residual',
            'training_boundary':'Only EEG heads and preprocessing fitted. No measured fMRI used; native TRIBE branch unchanged.',
            'correction_bound_is_not_accuracy_guarantee':True,
            'inference_mode':'Offline whole-content prediction, not causal real-time forecasting'},
        'data':previous['data'],'targets':previous['targets'],
        'evaluation':{'design':'Same eight crossed participant/topic-family folds and 1020 rows as v2',
            'status':report['evaluation_status'],'fresh_validation':False,
            'primary_candidate':'fusion','primary_comparator':'tribe_temporal',
            'success_point_estimate_met':passed,'primary_comparison':primary,
            'all_arms':report['metrics'],'all_comparisons':report['comparisons'],'all_seeds':report['seed_normalized_MAE'],
            'model_selection':'Deploy bounded residual chosen before scores, never the best-scoring arm or seed',
            'negative_control':control,'negative_control_scope':'Only correction EEG shuffled; TRIBE-only anchor remains trained on correctly paired EEG',
            'uncertainty':'Paired participant/lecture cluster resampling conditional on fitted models; only four lectures',
            'deployment':'Separately refitted on all eligible data after evaluation; training predictions are not validation'},
        'inference':{'entrypoint':'brain.content_pipeline.run(..., fusion_model=Path(...))',
            'formats':['video','audio','text','image','pdf'],'nonvideo_status':'Experimental transfer; absent blocks zeroed and reported',
            'execution':'Existing content-only pretrained inference plus local EEG head; no EEG required at inference',
            'native_cortical_predictions':'Copied unchanged with original clock; not measured participant fMRI',
            'five_format_checks':'Cached versions of one lesson, engineering checks only',
            'new_paid_requests':0},
        'limitations':[
            'V3 settings were motivated by v2 errors on these same lectures. This is exploratory reanalysis.',
            'Only four distinct lectures; repeated windows/viewers do not supply independent stimulus diversity.',
            'Same-content target agreement is often weak. The model cannot reconstruct unobserved person-specific responses.',
            'Receiver timestamps do not establish a physical sample clock. Contact quality, channel calibration and artifact handling remain unqualified.',
            'No new filtering, targets, rows or exclusions were introduced to improve the score.',
            'Fixing unstable scaling does not prove that features encode attention or learning.',
            'Improving native TRIBE fMRI requires a separate measured-fMRI evaluation; it was not tested here.'],
        'not_supported':previous['not_supported'],
        'privacy':'Fitted EEG weights remain local. No participant EEG or fitted weights sent to cloud.',
        'evidence':{'benchmark':'benchmark/report.json','audit':'audit.json','quality':'signal-quality.json',
            'qualification':'qualification.json','protocol':'protocol.json','model':'model/manifest.json'}}
    write_json(output/'model-card.json',card);write_json(Path('docs/brain_model_card_v3.json'),card)
    integration=read_json(Path('docs/brain_fusion_integration_v2.json'))
    integration.update(schema='perception-fusion-integration-v3',model_artifact=str((output/'model').resolve()),
        model_manifest_sha256=file_sha256(output/'model/manifest.json'),benchmark=str((output/'benchmark/report.json').resolve()),
        app_example=str((output/'video-example/app-result.json').resolve()),benchmark_success_target_met=passed,
        primary_normalized_MAE=metric,scientific_status='exploratory_revision_after_v2; inspect uncertainty and negative control',
        transfer='Absent representation/descriptor blocks are zeroed and reported; nonvideo use remains experimental transfer',
        legacy_selection_preserved=True)
    write_json(output/'integration.json',integration);write_json(Path('docs/brain_fusion_integration_v3.json'),integration)
    write_json(output/'reproduction.json',{'working_directory':str(Path.cwd()),
        'python':'/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python',
        'environment':{'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
        'commands_use_fresh_directories':[
            'python -m brain.fusion_v3_experiment run configs/brain_fusion_v3.json NEW/evaluation',
            'python -m brain.fusion_v3_audit NEW/evaluation NEW/audit',
            'python -m brain.fusion_v3_report NEW/evaluation NEW/benchmark',
            'python -m brain.fusion_v3_experiment deploy configs/brain_fusion_v3.json NEW/evaluation NEW/benchmark NEW/model',
            'python -m brain.fusion_qualify NEW/model .local/brain-runs NEW/qualification'],
        'cached_example':'python -m brain.fusion_inference --native .local/brain-runs/full-video-qualification-v1/tribe --model MODEL --output NEW_PREDICTION --duration 60.359375 --input-format video --features .local/brain-runs/full-video-qualification-v1/features',
        'app_JSON':'python -m brain.fusion_export NEW_PREDICTION NEW_PREDICTION/app-result.json'})
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.6f}</td></tr>' for k,v in metric.items())
    lo,hi=primary['reduction_interval_95']
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG fusion v3 results</title><style>body{{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#22303c}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:9px}}img{{width:100%}}.status{{padding:20px;background:#eef3f6}}a{{color:#235d84}}</style>
<body><p>EXPLORATORY REANALYSIS · FOUR LECTURES · V3</p><h1>Stable multimodal EEG prediction</h1>
<p class="status"><strong>The architecture runs.</strong> {html.escape(conclusion)}<br><strong>Native TRIBE fMRI outputs are unchanged; fMRI superiority is untested.</strong></p>
<p>The revised EEG head bounds feature scales and the influence of multimodal corrections. Existing audio, visual, language, SAM and TRIBE representations remain in the system. The standalone research project and failed v2 benchmark are preserved.</p>
<p>Primary EEG comparison: <strong>{100*primary['relative_error_reduction']:.2f}% relative error reduction</strong> versus the TRIBE-only EEG predictor. Descriptive 95% interval: {100*lo:.2f}% to {100*hi:.2f}%. Lower error is better; this is not a percentage accuracy score.</p>
<table><tr><th>All prespecified arms</th><th>Normalized MAE</th></tr>{rows}</table>
<p>All arms use identical supported rows and eight crossed participant/topic-family folds. The negative control shuffles only correction targets, not the frozen anchor. {html.escape(control)}</p>
<img src="benchmark/benchmark.png" alt="All prespecified benchmark arms">
<h2>Predetermined example</h2><p>First lecture alphabetically, lowest eligible participant. These are coarse channel summaries, not attention estimates or raw EEG.</p>
<img src="benchmark/predicted-versus-recorded.png" alt="Recorded and predicted channel summaries">
<h2>What remains unresolved</h2><p>Only four independent lessons, weak same-content agreement across viewers, unqualified physical EEG timing and contact quality. Additional paired data and fresh evaluation are needed; no model architecture removes those limitations.</p>
<p><a href="model-card.json">Model card</a> · <a href="benchmark/report.json">Complete benchmark</a> · <a href="signal-quality.json">Signal audit</a> · <a href="audit.json">Replay audit</a> · <a href="qualification.json">Five-format checks</a> · <a href="integration.json">Integration contract</a> · <a href="reproduction.json">Reproduction</a> · <a href="video-example/app-result.json">Video example</a></p>
<p>Local-only bundle contains fitted EEG weights. No attention, retention, demographic or quiz predictions.</p></body></html>'''
    (output/'index.html').write_text(page)
    files={str(p.relative_to(output)):file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'handoff-manifest.json',{'schema':'perception-fusion-handoff-v3','files_sha256':files,
        'local_only_contains_EEG_weights':True,'contains_raw_participant_EEG':False,
        'source_code_sha256':{p.name:file_sha256(p) for p in Path('brain').glob('fusion*.py')}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.runs,a.output)
