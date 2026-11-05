"""Qwen upgrade diagnostic with fixed anchors; never emits training labels."""
import argparse
import json
from pathlib import Path
import time

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from . import QUALITIES
from .content_teacher_v2 import messages, response_from_scores, LETTERS, POLICY
from .content_teacher_v2_probes import PAIRS, CHECK_QUALITIES, MIN_DIFFERENCE, digest
from .labels import reconcile_synthetic
from .mlx_teacher import runtime
from .acquire_qwen_teacher import REPO, REVISION
from .rubrics import RUBRICS


def verify_identity(checkpoint):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint/'acquisition.json').read_text())
    if not metadata['complete'] or metadata['repo'] != REPO or metadata['revision'] != REVISION:
        raise ValueError('pinned Qwen checkpoint required')
    from .acquire_qwen_teacher import FILES
    if set(metadata['files']) != set(FILES):
        raise ValueError('checkpoint inventory mismatch')
    for name, expected in metadata['files'].items():
        guard(checkpoint)
        if (checkpoint/name).resolve().parent != checkpoint.resolve() or file_sha256(checkpoint/name) != expected:
            raise ValueError('checkpoint changed')
    return metadata


def render_prompt(tokenizer, msg, prefill):
    rendered = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer.encode(rendered+prefill, add_special_tokens=False)
    if len(encoded)>4096:
        raise ValueError('token limit exceeded')
    return encoded


def run(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    guard(output); runtime()
    protocol = json.loads((output/'protocol.json').read_text())
    packets = Path(protocol['packet_source'])
    if (file_sha256(packets) != protocol['packet_sha256']
            or file_sha256(POLICY) != protocol['original_prompt_policy_sha256']
            or protocol['assistant_prefill'] != 'Answer: ' or protocol['training_eligible']
            or protocol['repo'] != REPO or protocol['revision'] != REVISION
            or protocol['enable_thinking'] is not False
            or protocol['runner_sha256'] != file_sha256(Path(__file__))):
        raise ValueError('fixed diagnostic identity required')
    if (output/'responses.json').exists():
        raise ValueError('preserve previous diagnostic output')
    verify_identity(checkpoint)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    mx.random.seed(17)
    model, tokenizer = load(str(checkpoint), tokenizer_config={'trust_remote_code': False}, trust_remote_code=False)
    forms = {letter: [(text, tokenizer.encode(text, add_special_tokens=False)) for text in (letter, ' '+letter)] for letter in LETTERS}
    if any(len(ids) != 1 for rows in forms.values() for _,ids in rows):
        raise ValueError('single-token answer forms required')
    ids = [tokens[0] for rows in forms.values() for _,tokens in rows]
    results = []
    for record in json.loads(packets.read_text()):
        for rubric in RUBRICS:
            raw = {}; hashes = {}; started = time.monotonic()
            for quality in QUALITIES:
                guard(output)
                msg = messages(rubric, quality, record['packet'])
                encoded = render_prompt(tokenizer,msg,protocol['assistant_prefill'])
                generator = generate_step(mx.array(encoded), model, max_tokens=1, prefill_step_size=256,
                    logits_processors=[lambda tokens, logits: logits.astype(mx.float32)])
                try:
                    _, logprobs = next(generator)
                    selected = logprobs[mx.array(ids)].tolist()
                finally:
                    generator.close()
                raw[quality] = {letter: [{'text': text, 'token_id': tokens[0], 'log_probability': float(selected[2*i+j])}
                    for j,(text,tokens) in enumerate(forms[letter])] for i,letter in enumerate(LETTERS)}
                hashes[quality] = digest({'messages': msg, 'assistant_prefill': protocol['assistant_prefill']})
                del logprobs; mx.clear_cache()
            parsed = response_from_scores(raw, record['packet'])
            results.append({'id':record['id'], 'rubric':rubric, 'raw_scores':raw, 'response':parsed,
                'prompt_sha256':hashes, 'elapsed_seconds':time.monotonic()-started})
            (output/'responses.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            print(record['id']+' '+rubric+': '+str(sum(v['distribution'] is not None for v in parsed.values()))+'/5 scored',flush=True)
    lookup = {(r['id'],r['rubric']):r['response'] for r in results}
    comparisons = []
    for low,high in PAIRS:
        for quality in CHECK_QUALITIES:
            scores = {}
            for name in (low,high):
                consensus = reconcile_synthetic(*(lookup[name,r][quality]['distribution'] for r in RUBRICS))
                scores[name] = None if not consensus['accepted'] else sum(p*x for p,x in zip(consensus['distribution'],(0,2.5,5,7.5,10)))
            a,b = scores[low],scores[high]
            difference = None if a is None or b is None else b-a
            comparisons.append({'pair':[low,high],'quality':quality,'scores':scores,'difference':difference,
                'passed':difference is not None and difference>=MIN_DIFFERENCE})
    missing = all(lookup['missing_content',r][q]['distribution'] is None for r in RUBRICS for q in QUALITIES)
    report = {'complete':True,'training_eligible':False,'teacher_repo':REPO,'teacher_revision':REVISION,'enable_thinking':False,'diagnostics_are_reused_development_examples':True,
        'protocol_sha256':file_sha256(output/'protocol.json'),'runner_sha256':file_sha256(Path(__file__)),
        'checkpoint_receipt_sha256':file_sha256(checkpoint/'acquisition.json'),
        'responses_sha256':file_sha256(output/'responses.json'),'comparisons':comparisons,
        'missing_content_abstention':missing,'gate_passed':missing and all(r['passed'] for r in comparisons),
        'accepted_quality_responses':sum(v['distribution'] is not None for r in results for v in r['response'].values()),
        'total_quality_responses':len(results)*len(QUALITIES),'participant_data_sent_to_cloud':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gate_passed':report['gate_passed'],'accepted':report['accepted_quality_responses']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.checkpoint,a.output)
