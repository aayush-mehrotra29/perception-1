"""Concept identifiers from authenticated, verbatim content facts, not student memory."""
import hashlib
import json
from pathlib import Path

from brain.auxiliary_windows import authenticated_child
from brain.meta_features import ContentSpan,SemanticResult,validate_semantics


def collect(directory,source_sha256):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!='perception-lesson-semantics-v1' or manifest['source']['sha256']!=source_sha256:
        raise ValueError('semantic concept source mismatch')
    occurrences=[];seen=set()
    for record in manifest['batches']:
        path=authenticated_child(directory,record['manifest'],record['sha256'])
        batch=json.loads(path.read_text())
        if batch['source_sha256']!=source_sha256:raise ValueError('concept batch source mismatch')
        spans=[ContentSpan.model_validate(s) for s in batch['spans']]
        facts=SemanticResult.model_validate(batch['result']);validate_semantics(facts,spans)
        lookup={s.id:s for s in spans}
        for fact in facts.facts:
            if fact.kind!='topic_term':continue
            span=lookup[fact.span_id];key=(fact.quote,span.start,span.stop)
            if key in seen:continue
            seen.add(key)
            occurrences.append({'id':'concept-'+hashlib.sha256(fact.quote.encode()).hexdigest()[:20],
                'text':fact.quote,'start':span.start,'stop':span.stop,'source_span_id':span.id,
                'source_sha256':source_sha256,'fact_manifest_sha256':record['sha256']})
    return sorted(occurrences,key=lambda c:(c['start'],c['id']))

