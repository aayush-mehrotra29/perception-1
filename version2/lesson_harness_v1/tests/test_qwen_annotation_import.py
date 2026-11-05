import copy
import hashlib
import json
from pathlib import Path

import pytest

from lesson_harness_v1 import QUALITIES
from lesson_harness_v1 import qwen_annotation_import as importer
from lesson_harness_v1.content_teacher_v2 import messages,response_from_scores
from lesson_harness_v1.content_teacher_v2_probes import digest
from lesson_harness_v1.rubrics import RUBRICS
from .test_annotation_import import fixture
from .test_content_teacher_v2 import scores


def qwen_fixture():
    records,_,human,windows=fixture()
    packet=records[0]['packet'];packet['evidence'][0]['kind']='transcript'
    rows=[]
    for rubric in RUBRICS:
        raw=scores()
        rows.append({'id':records[0]['id'],'rubric':rubric,'method':'qwen-content-ordinal-v1',
            'packet_sha256':digest(packet),'status':'valid','raw_scores':raw,
            'prompt_sha256':{q:digest({'messages':messages(rubric,q,packet),'assistant_prefill':'Answer: '}) for q in QUALITIES},
            'response':response_from_scores(raw,packet)})
    return records,rows,human,windows


def test_reconciles_qwen_without_replacing_human_ground_truth():
    records,rows,human,windows=qwen_fixture()
    result,audit=importer.reconcile(records,rows,human,windows,'f'*64)
    assert result[0]==human[0] and len(result)==5
    assert all(o.label.origin=='synthetic' for o in result[1:])
    assert audit[0]['reason']=='existing_human_label_same_scope'


def test_tampered_prompt_or_repaired_scores_are_rejected():
    records,rows,human,windows=qwen_fixture()
    edited=copy.deepcopy(rows);edited[0]['prompt_sha256'][QUALITIES[0]]='0'*64
    with pytest.raises(ValueError,match='prompt identity'):
        importer.reconcile(records,edited,human,windows,'f'*64)
    edited=copy.deepcopy(rows);edited[0]['response'][QUALITIES[1]]['distribution']=[0,0,0,0,1]
    with pytest.raises(ValueError,match='differs'):
        importer.reconcile(records,edited,human,windows,'f'*64)


def test_abstention_and_rubric_disagreement_never_become_training_targets():
    records,rows,human,windows=qwen_fixture()
    rows[1]['raw_scores']=scores(high=True)
    rows[1]['response']=response_from_scores(rows[1]['raw_scores'],records[0]['packet'])
    result,audit=importer.reconcile(records,rows,human,windows,'f'*64)
    assert result==human and all(not row['accepted'] for row in audit)


def test_qwen_amendment_keeps_numerical_defaults_and_separate_identity():
    root=Path(importer.__file__).parent
    old=json.loads((root/'config.json').read_text());new=json.loads((root/'qwen_config.json').read_text())
    for name in ['training','architecture','preprocessing','eeg_auxiliary','arms']:
        assert new[name]==old[name]
    assert new['synthetic_labels']['teacher']==importer.REPO
    assert old['synthetic_labels']['teacher']!=new['synthetic_labels']['teacher']
