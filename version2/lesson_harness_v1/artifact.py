"""Deployment qualification before any paid inference. No partial/random-model fallback."""
import json
from pathlib import Path

from brain.worker_contract import file_sha256
from . import QUALITIES, VERSION
from .features import WIDTH, load_transforms


def child(root, name, digest):
    root = Path(root).resolve();path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha256(path)!=digest:
        raise ValueError('artifact path/checksum mismatch')
    return path


def preflight(directory):
    """Checks manifest, training/benchmark evidence and every numeric member hash.

    This does not equate an exploratory benchmark with human scientific validation.
    A final artifact must record the fitted status of each head, even if blocked.
    """
    root = Path(directory)
    meta = json.loads((root/'manifest.json').read_text())
    if (meta.get('schema')!='perception-lesson-deployment-v1' or meta.get('model_version')!=VERSION
            or meta.get('deployment_status')!='fitted' or meta.get('input_width')!=WIDTH
            or meta.get('qualities')!=list(QUALITIES) or meta.get('context_seconds')!=44
            or meta.get('query_step_seconds')!=1 or meta.get('synthetic_fixture',True)):
        raise ValueError('fitted real-data lesson artifact required')
    if not isinstance(meta.get('reference_population'),str) or not meta['reference_population'].strip():
        raise ValueError('explicit reference population required')
    if set(meta.get('heads',{}))!=set(QUALITIES):raise ValueError('all five head statuses required')
    for name,head in meta['heads'].items():
        if head.get('status') not in {'fitted','blocked'}:raise ValueError('unknown head status')
        if head['status']=='fitted':
            if (not head.get('supervision_origins') or set(head['supervision_origins'])-{'human','synthetic'}
                    or head.get('training_labels',0)<1 or head.get('validation_status') not in {'exploratory','human_evaluated'}):
                raise ValueError('fitted head requires supervision and evaluation provenance')
        elif not head.get('reason'):raise ValueError('blocked head requires reason')
    if not any(h['status']=='fitted' for h in meta['heads'].values()):raise ValueError('no fitted heads')
    for name in ['config.json','training.json','benchmark.json','label-audit.json']:
        child(root,name,meta['files'][name])
    training=json.loads((root/'training.json').read_text())
    benchmark=json.loads((root/'benchmark.json').read_text())
    if not training.get('complete') or training.get('purpose')!='full_data_deployment' or not benchmark.get('complete'):
        raise ValueError('complete deployment training and benchmark required')
    if [x['seed'] for x in meta.get('ensemble',[])]!=[17,29,43]:raise ValueError('complete fixed seed ensemble required')
    for member in meta['ensemble']:
        path=child(root,member['manifest'],member['sha256'])
        network=json.loads(path.read_text())
        if network.get('schema')!='perception-lesson-network-v1' or network.get('input_width')!=WIDTH:
            raise ValueError('network schema/input width mismatch')
        child(path.parent,'weights.npz',network['weights_sha256'])
    transform_manifest=child(root,'transforms/manifest.json',meta['files']['transforms/manifest.json'])
    load_transforms(transform_manifest.parent)
    return meta

