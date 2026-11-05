import numpy as np
from lesson_harness_v1.eeg_quality_frontal import contact_support,flatline_mask
from lesson_harness_v1.eeg_auxiliary import prepare
from lesson_harness_v1.annotation_packets import make_packet
from lesson_harness_v1.quality_import import select_screened


def test_frontal_screen_only_uses_actual_frontal_contacts():
    all_codes=np.array([[4,1,1,4],[4,1,1,4],[4,2,1,4]])
    assert contact_support([50,150],[0,100,200],all_codes[:,[1,2]]).tolist()==[True,False]
    values=np.array([[1,2],[1,3],[1,4],[1,5]],float)
    assert flatline_mask(values,np.arange(4),4).all()


def test_frontal_targets_preserve_channel_names_and_width():
    windows=[{'lesson':'a','source_sha256':'a'*64,'query_seconds':45+i*5,'context_start':1+i*5} for i in range(2)]
    eeg={'person':np.array([1,1]),'lesson':np.array(['a','a']),'point':np.array([0,1]),'target':np.zeros((2,8))}
    qualified={(1,'a',w['query_seconds']):{'media_sha256':'a'*64,'target':np.array([700,701,2+i,3+i])} for i,w in enumerate(windows)}
    selected,excluded=select_screened(eeg,windows,qualified,['AF7','AF8'])
    assert selected['target'].shape==(2,4) and excluded==[]
    aux=prepare(selected,[0,1]);assert aux['target'].shape==(2,4)
    packet=make_packet(windows[0],selected['target'][0],[],['AF7','AF8'])
    summary=packet['evidence'][-1]
    assert summary['channel_order']==['AF7','AF8']
    assert summary['mean_received_microvolts']==[700,701]
    assert summary['log1p_received_variance_microvolts_squared']==[2,3]
