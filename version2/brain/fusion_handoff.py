"""Assemble a local-only model, evidence, figures, and app example after validation."""
import argparse
from datetime import datetime, timezone
import html
from pathlib import Path
import shutil

from .fusion_data import read_json, write_json
from .edugage_targets import guard
from .worker_contract import file_sha256


def handoff(runs, output):
    runs, output = Path(runs), Path(output)
    guard(output)
    report = read_json(runs / "fusion-v2-benchmark/report.json")
    audit = read_json(runs / "fusion-v2-audit/audit.json")
    qualification = read_json(runs / "fusion-v2-qualification/qualification.json")
    model = runs / "fusion-v2-deployment"
    model_meta = read_json(model / "manifest.json")
    if (audit["status"] != "passed" or qualification["status"] != "passed"
            or qualification["model_manifest_sha256"] != file_sha256(model / "manifest.json")
            or audit["evaluation_manifest_sha256"] != report["evaluation_manifest_sha256"]):
        raise ValueError("complete matching validation required")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(model, output / "model")
    shutil.copytree(runs / "fusion-v2-benchmark", output / "benchmark")
    shutil.copy2(runs / "fusion-v2-audit/audit.json", output / "audit.json")
    shutil.copy2(runs / "fusion-v2-audit/metric-audit.json", output / "metric-audit.json")
    shutil.copy2(runs / "fusion-v2-qualification/qualification.json", output / "qualification.json")
    shutil.copy2(runs / "fusion-v2-prepared/protocol.json", output / "protocol.json")
    shutil.copy2(runs / "fusion-v2-pilot/runtime.json", output / "runtime-pilot.json")
    shutil.copy2(Path("configs/brain_fusion_runtime_v2.json"), output / "local-runtime.json")
    for name in ("fusion-v2-tests-final-base.log", "fusion-v2-tests-torch.log", "fusion-v2-runtime-resource-snapshot.json"):
        shutil.copy2(runs / name, output / name)
    shutil.copytree(runs / "fusion-v2-qualification/video/prediction", output / "video-example")
    prepared = read_json(runs / "fusion-v2-prepared/manifest.json")
    write_json(output / "eligibility.json", {"prior_viewing_exclusions": prepared["prior_exclusions"],
        "additional_sequence_exclusions": prepared["sequence_exclusions"],
        "prepared_manifest_sha256": file_sha256(runs / "fusion-v2-prepared/manifest.json")})
    native = read_json(runs / "full-video-qualification-v1/tribe/manifest.json")
    write_json(output / "frozen-components.json", {k: native[k] for k in
        ("checkpoint_revision", "checkpoint_sha256", "upstream_config_sha256", "tribe_version",
         "observed_encoder_revisions", "runtime_versions", "worker_source_sha256")})
    comparison = report["comparisons"]["tribe_temporal"]
    passed = report["success_target_point_estimate_met"]
    conclusion = ("The prespecified point-estimate success target was met." if passed else
                  "The prespecified success target was not met.")
    if passed and not report["primary_interval_excludes_zero_in_favorable_direction"]:
        conclusion += " The descriptive interval spans zero; this is only a preliminary improvement."
    errors = {name: metric["primary_normalized_MAE"] for name, metric in report["metrics"].items()}
    control_interpretation = ("The shuffled-target control matched or outperformed fusion. These results do not establish that correct within-viewing temporal EEG pairing improves prediction."
        if errors["shuffled_fusion"] <= errors["fusion"] else
        "Fusion had lower point error than the shuffled-target control; use the reported paired interval to assess uncertainty. This is not evidence of attention or learning prediction.")
    tribe_interpretation = ("Adding TRIBE representations did not improve the point estimate over the no-TRIBE arm."
        if errors["without_tribe"] <= errors["fusion"] else
        "Adding TRIBE representations lowered point error relative to the no-TRIBE arm; uncertainty is reported separately.")
    card = {
        "schema": "perception-brain-model-card-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_runs": True, "EEG_benchmark_conclusion": conclusion,
        "measured_fmri_superiority_tested": False,
        "scope": "Competition-only exploratory composite model; standalone neuroscience project is unchanged",
        "architecture": {
            "frozen": ["Llama 3.2", "W2v-BERT", "V-JEPA2", "native TRIBE cortical predictor"],
            "EEG_trainable": "Three-seed ensemble of 16,296-parameter residual temporal convolutional networks",
            "temporal_input": "88 ordered coverage-supported bins at 2 Hz over a 44-second context",
            "preprocessing": "Training-fold scaling and PCA capped at 16 dimensions per block, fitted to unique training content",
            "auxiliary": "SAM visual, SAM Audio, semantic, sensory, and progress descriptors enter after temporal pooling",
            "variance_output": "Positive softplus parameterization; no post-hoc clipping",
            "training_boundary": "Only the small EEG model and its transforms were trained; the fMRI branch is unchanged",
            "inference_mode": "Whole-lesson offline prediction; encoder context may be bidirectional, so this is not causal real-time forecasting"},
        "data": {"source": "Four previously audited EduGage lectures", "people": 16,
            "scheduled_viewings": 32, "observed_supported_viewings": 31,
            "missing_EEG": "Participant 2, xray_diffraction", "rows": 1020, "unique_content_windows": 145,
            "independent_content_items": 4, "additional_sequence_exclusions": 0,
            "effective_sample_warning": "Overlapping contexts and repeated participant targets do not create 1,020 independent lessons"},
        "targets": {"names": report["targets"], "channels": ["TP9", "AF7", "AF8", "TP10"],
            "context_seconds": 44, "query_hop_seconds": 5,
            "units": "Training-source channel means and log1p population variances; no calibrated waveform claim",
            "alignment": "Existing coarse receiver-to-reported-playback association; physical sample synchronization is unqualified",
            "signal_quality": "Channel calibration and artifact/contact quality remain unqualified",
            "not_individualized": "Same content produces the same ensemble mean; no learned demographics or person-specific brain model"},
        "evaluation": {"design": "Eight crossed participant-group/topic-family folds, no shared participants or lessons",
            "status": "Exploratory reanalysis of previously evaluated lectures",
            "primary": "Equal-lecture, equal-viewing MAE across eight targets normalized by training-fold target SD",
            "success_rule": "At least 5% reduction versus TRIBE-only temporal comparator and better than training mean",
            "success_point_estimate_met": passed, "primary_comparison": comparison,
            "all_arms": report["metrics"], "all_seeds": report["seed_normalized_MAE"],
            "all_comparisons": report["comparisons"], "temporal_networks_reloaded_and_forecasts_reproduced": audit["networks_replayed"],
            "ridge_models_refitted_and_replayed": audit["ridge_models_refitted_and_replayed"],
            "negative_control_interpretation": control_interpretation,
            "incremental_TRIBE_interpretation": tribe_interpretation,
            "uncertainty": "Paired lecture/participant resampling conditional on fixed fits; four lectures cannot establish population generalization",
            "deployment": "Separately fitted on all eligible rows after evaluation; its training predictions were never validation"},
        "inference": {"entrypoint": "brain.content_pipeline.run(..., fusion_model=Path(...))",
            "legacy_selection_preserved": True, "formats": ["video", "audio", "text", "image", "pdf"],
            "nonvideo_status": "Experimental transfer from audiovisual training, with explicit absent-modality masks",
            "presentation_clock": "Text requires reading pace; documents require page durations; these are assumptions, not measured gaze",
            "nonvideo_auxiliary": "Unavailable window-level SAM/semantic/sensory descriptors are masked; only video has the qualified auxiliary assembly",
            "five_format_engineering_checks": "Five versions of one original velocity lesson, no measured brain or behavioral ground truth",
            "output": "Versioned JSON/NPZ; unsupported EEG values export as null; native cortical arrays and clocks are unchanged",
            "execution": "Cloud pretrained content inference plus local fitted EEG head; not browser-only model execution",
            "local_runtime": "PyTorch 2.14.1 via existing research interpreter or BRAIN_TORCH_PYTHON; research environment not modified",
            "new_paid_inference_requests_in_sprint": 0},
        "not_supported": ["Attention, distraction, cognitive load, learning, retention or quiz scores",
            "Causal effects of video features", "Demographic student personas", "Raw EEG waveforms",
            "Joint training on measured EEG and fMRI", "Outperforming TRIBE on measured fMRI",
            "Accuracy on silent reading or audio-only responses", "Clinical or educational decisions about individual students"],
        "limitations_and_next_research": ["Independent lessons and participants are needed for fresh validation",
            "Confirm physical EEG timing, channel units and contact/artifact quality before neural claims",
            "A strong shuffled-label control would weaken any claim that temporal content relationships were learned",
            "Behavioral attention or quiz outputs require separate audited labels and evaluation"],
        "privacy": "Local-only bundle contains fitted EEG weights; do not include it in cloud content requests",
        "evidence": {"benchmark": "benchmark/report.json", "audit": "audit.json", "qualification": "qualification.json",
            "protocol": "protocol.json", "model": "model/manifest.json", "video_example": "video-example/app-result.json"}}
    write_json(output / "model-card.json", card)
    write_json(Path("docs/brain_model_card_v2.json"), card)
    integration = read_json(Path("docs/brain_fusion_integration_v2.json"))
    integration.update(status="fitted_and_engineering_verified; see benchmark for scientific outcome",
        benchmark_success_target_met=passed,
        primary_normalized_MAE={name: errors[name] for name in ("fusion", "tribe_temporal", "mean")},
        scientific_status="exploratory_prototype; no demonstrated improvement" if not passed else "exploratory_reanalysis; inspect uncertainty and controls",
        model_artifact=str((output / "model").resolve()),
        model_manifest_sha256=file_sha256(output / "model/manifest.json"),
        benchmark=str((output / "benchmark/report.json").resolve()),
        app_example=str((output / "video-example/app-result.json").resolve()))
    Path("docs/brain_fusion_integration_v2.json").write_text(__import__('json').dumps(integration, indent=2) + "\n")
    write_json(output / "integration.json", integration)
    write_json(output / "reproduction.json", {
        "working_directory": str(Path.cwd()), "environment": {"PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"},
        "python": "/Users/Aayush/Workspace/Nueroscience Research/.venv/bin/python",
        "commands_use_fresh_output_directories": [
            "python -m brain.fusion_experiment prepare .local/brain-corpus-execution-v2.json configs/brain_fusion_v2.json NEW/prepared",
            "python -m brain.fusion_experiment pilot NEW/prepared NEW/pilot",
            "python -m brain.fusion_experiment run NEW/prepared NEW/pilot NEW/evaluation",
            "python -m brain.fusion_audit NEW/prepared NEW/evaluation NEW/audit",
            "python -m brain.fusion_report NEW/evaluation NEW/benchmark",
            "python -m brain.fusion_experiment deploy NEW/prepared NEW/evaluation NEW/benchmark NEW/model",
            "python -m brain.fusion_qualify NEW/model .local/brain-runs NEW/qualification"],
        "cached_video_example": "python -m brain.fusion_inference --native .local/brain-runs/full-video-qualification-v1/tribe --model MODEL --output NEW_PREDICTION --duration 60.359375 --input-format video --features .local/brain-runs/full-video-qualification-v1/features",
        "app_JSON": "python -m brain.fusion_export NEW_PREDICTION NEW_PREDICTION/app-result.json"})
    render_report(card, output)
    files = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "handoff-manifest.json", {"schema": "perception-fusion-handoff-v2", "files_sha256": files,
        "local_only_contains_EEG_weights": True, "contains_raw_participant_EEG": False,
        "source_code_sha256": {p.name: file_sha256(p) for p in sorted(
            list(Path("brain").glob("fusion*.py")) + [Path("brain") / name for name in
            ("pipeline.py", "content_pipeline.py", "bundle.py", "worker_contract.py", "modal_client.py")])}})
    return card


def render_report(card, output):
    e = html.escape
    metrics = card["evaluation"]["all_arms"]
    rows = "".join(f"<tr><td>{e(k)}</td><td>{v['primary_normalized_MAE']:.5f}</td><td>{v['normalized_RMSE']:.5f}</td><td>{v['negative_log1p_variance_predictions']}</td></tr>" for k, v in metrics.items())
    c = card["evaluation"]["primary_comparison"]
    lo, hi = c["reduction_interval_95"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fusion v2 — EEG transfer benchmark</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#20272d;background:#fafafa;max-width:1050px;margin:48px auto;padding:0 24px}}h1{{font-size:34px;line-height:1.15}}h2{{margin-top:36px;font-size:22px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce0e3}}th{{font-size:13px}}img{{width:100%;background:white}}.status{{padding:18px;background:#edf1f4;border-left:4px solid #346b99}}code{{background:#edf1f4;padding:2px 5px}}a{{color:#225c89}}small{{color:#53616d}}
</style><body><small>EXPLORATORY REANALYSIS · FOUR LECTURES · FUSION v2</small>
<h1>Content-to-EEG prediction with frozen multimodal and TRIBE representations</h1>
<p class="status"><strong>Architecture: runs.</strong> {e(card['EEG_benchmark_conclusion'])}<br><strong>Measured-fMRI superiority: not tested.</strong> The native TRIBE cortical branch is unchanged.</p>
<p>The model combines frozen language, audio and visual representations with TRIBE-predicted cortical features. A small local temporal network learns eight EEG summary outputs; window-level SAM and content descriptors enter after temporal pooling.</p>
<img src="benchmark/architecture.png" alt="Architecture showing frozen content and TRIBE branches feeding the EEG-trained temporal network">
<h2>Prespecified comparison</h2><p>Fusion versus the matched TRIBE-only temporal predictor: <strong>{100*c['relative_error_reduction']:.2f}% relative error reduction</strong>; descriptive 95% interval <strong>{100*lo:.2f}% to {100*hi:.2f}%</strong>. Negative reductions mean worse error. Success required ≥5% reduction and improvement over the training-mean baseline.</p>
<table><thead><tr><th>Arm</th><th>Normalized MAE ↓</th><th>Normalized RMSE ↓</th><th>Negative variance outputs</th></tr></thead><tbody>{rows}</tbody></table>
<p>All arms use the same 1,020 eligible rows, eight crossed participant/topic-family folds and training-fold target scales. Each temporal arm averages seeds 17, 29 and 43. No best-seed or endpoint selection was used.</p>
<img src="benchmark/benchmark.png" alt="All six prespecified arms and normalized errors">
<img src="benchmark/comparisons.png" alt="Descriptive paired confidence intervals for all five prespecified comparisons">
<h2>Predetermined example</h2><p>Alphabetically first lecture and lowest eligible participant, chosen without inspecting prediction accuracy. Gaps remain gaps. These are channel summary features, not calibrated raw EEG waveforms or attention scores.</p>
<img src="benchmark/predicted-versus-recorded.png" alt="Recorded EEG summaries compared with held-out fusion and TRIBE-derived predictions">
<h2>What the result can establish</h2><p>This tests EEG transfer on four already evaluated lectures and 31 observed viewings from 16 people. Repeated participants and overlapping windows are not independent content items. The confidence interval is descriptive, conditional on fitted models. Coarse playback timing, signal quality and channel calibration remain unqualified.</p>
<p>The temporal comparator has the same network family and training budget, but different available information: non-TRIBE input slots and all auxiliary descriptors are masked. The shuffled-target arm tests how much apparent performance can survive broken content–target alignment.</p>
<p><strong>Control result:</strong> {e(card['evaluation']['negative_control_interpretation'])} {e(card['evaluation']['incremental_TRIBE_interpretation'])}</p>
<h2>Integration and evidence</h2><p>The exported head is separately fitted on all eligible training rows. It needs no EEG at inference. A local Python backend runs the EEG ensemble; the browser receives arrays/JSON. Large pretrained components use the existing content-only cloud pipeline.</p>
<ul><li><a href="model-card.json">Complete model card</a> · <a href="protocol.json">Frozen protocol</a></li>
<li><a href="benchmark/report.json">All metrics, comparisons and seeds</a> · <a href="audit.json">Independent artifact audit</a></li>
<li><a href="qualification.json">Five-format execution checks</a> · <a href="video-example/app-result.json">Verified video output</a></li>
<li><a href="integration.json">App integration contract</a> · <a href="reproduction.json">Reproduction commands</a></li></ul>
<p><small>Local-only handoff: fitted EEG weights remain local. No participant EEG or fitted EEG weights were sent to cloud inference. No attention, retention, persona, or quiz predictions are included.</small></p></body></html>"""
    (output / "index.html").write_text(page)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("runs", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); handoff(a.runs, a.output)
