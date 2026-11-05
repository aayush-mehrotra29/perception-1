"""Gated local Qwen content labels. Synthetic proxies, never observed student states."""
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
from .qwen_teacher_diagnostic import verify_identity, render_prompt
from .rubrics import RUBRICS


def run(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    guard(output); runtime()
    protocol = json.loads((output/'protocol.json').read_text())
    packets = Path(protocol['packet_source'])
    if (file_sha256(packets) != protocol['packet_sha256']
            or file_sha256(POLICY) != protocol['original_prompt_policy_sha256']
            or protocol['assistant_prefill'] != 'Answer: ' or not protocol['training_eligible']
            or protocol['repo'] != REPO or protocol['revision'] != REVISION
            or protocol['enable_thinking'] is not False
            or protocol['runner_sha256'] != file_sha256(Path(__file__))):
        raise ValueError('fixed diagnostic identity required')
    if (output/'responses.json').exists():
        raise ValueError('preserve previous diagnostic output')
    diagnostic = Path(protocol['diagnostic_report'])
    gate = json.loads(diagnostic.read_text())
    if (file_sha256(diagnostic) != protocol['diagnostic_report_sha256'] or not gate['gate_passed']
            or not gate['complete'] or gate['teacher_repo'] != REPO or gate['teacher_revision'] != REVISION
            or gate['checkpoint_receipt_sha256'] != file_sha256(checkpoint/'acquisition.json')
            or gate['runner_sha256'] != file_sha256(Path(__file__).with_name('qwen_teacher_diagnostic.py'))):
        raise ValueError('passing diagnostic required before corpus labeling')
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
    if len(set(ids)) != 12:
        raise ValueError('ordinal answer token forms overlap')
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
                'method':'qwen-content-ordinal-v1','packet_sha256':digest(record['packet']),'status':'valid',
                'prompt_sha256':hashes, 'elapsed_seconds':time.monotonic()-started})
            (output/'responses.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            print(record['id']+' '+rubric+': '+str(sum(v['distribution'] is not None for v in parsed.values()))+'/5 scored',flush=True)
    report = {'schema':'perception-qwen-content-teacher-v1','complete':True,
        'repo':REPO,'revision':REVISION,'seed':17,'backend':'mlx-affine-4bit-group64',
        'method':'qwen-content-ordinal-v1','enable_thinking':False,'assistant_prefill':'Answer: ',
        'packets_sha256':file_sha256(packets),'responses_sha256':file_sha256(output/'responses.json'),
        'checkpoint_receipt_sha256':file_sha256(checkpoint/'acquisition.json'),
        'checkpoint':str(checkpoint.resolve()),'runner_sha256':file_sha256(Path(__file__)),
        'validated_responses':len(results),'total_responses':len(results),
        'participant_data_sent_to_cloud':False,'diagnostic_report_sha256':file_sha256(diagnostic),
        'diagnostic_report':str(diagnostic.resolve()),'protocol_sha256':file_sha256(output/'protocol.json'),
        'probabilities_calibrated_to_humans':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.checkpoint,a.output)
