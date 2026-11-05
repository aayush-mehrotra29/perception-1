"""Bounded-memory, content-only adapter for the frozen endpoint EEG model.

Predicts a 0.5--8 Hz group response, not raw full-band or an individual's EEG.
Source EEG is never read. The ALICE training transforms are never refitted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import time
import wave
import importlib.util
import os
import numpy as np
from brain.worker_contract import file_sha256
from brain.edugage_targets import guard

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/Volumes/EEG Research Workspace/raw/wav2vec2-base-960h-22aad52-2026-10-24')
RECEIPT = '24a602de69b3e8d784a3fc66a059f69158f4e3d7d362ba0db0a80f1f5c4eac86'
PCA = Path('/Volumes/EEG Research Workspace/runs/alice-native-design-v1-2026-10-27/pca_state.npz')
PCA_SHA = 'e8407aa8af9579d6123c64ba171ec2d7b8be145c68f0200fda8f8abaa09ed584'

def read(p): return json.loads(Path(p).read_text())
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def project(x,p):
    return (np.clip((x-p['mean'])/p['scale'],-5,5)-p['center'])@p['components'].T

class EndpointPredictor:
    """Authenticate and load three fixed local models once; no cwd dependence."""
    def __init__(self,run):
        from speech_eeg.endpoint import load_endpoint
        from speech_eeg.cohort_data import SLOTS
        self.run=Path(run);lock=read(self.run/'forecast-lock.json');c=read(self.run/'config.json')
        for key,name in [('config_sha256','config.json'),('selection_sha256','development-selection.json'),('transforms_sha256','transforms.npz')]:
            if file_sha256(self.run/name)!=lock[key]:raise ValueError('endpoint artifact changed')
        base=Path(c['base_run']);base=base if base.is_absolute() else ROOT/base
        if file_sha256(base/'forecast-lock.json')!=read(self.run/'started.json')['base_lock_sha256']:raise ValueError('base lock changed')
        oldlock=read(base/'forecast-lock.json');old=read(base/'config.json')
        if file_sha256(base/'config.json')!=oldlock['config_sha256']:raise ValueError('base configuration changed')
        self.state=load(self.run/'transforms.npz');self.slots=SLOTS
        self.weight=read(self.run/'development-selection.json')['full']['weight']
        if self.weight!=.2:raise ValueError('fixed neural blend changed')
        r=next(r for r in oldlock['ridges'] if r['name']=='acoustic_ridge_labels8')
        if file_sha256(base/r['file'])!=r['sha256']:raise ValueError('acoustic model changed')
        self.ridge=load(base/r['file']);self.models=[]
        for seed in (17,29,43):
            r=next(r for r in lock['models'] if r['block']=='full' and r['seed']==seed)
            if file_sha256(self.run/r['file'])!=r['sha256']:raise ValueError('endpoint weights changed')
            self.models.append(load_endpoint(self.run/r['file'],len(SLOTS['full']),40,old['architecture']))

    def predict(self,features):
        from speech_eeg.inference import validate_features
        from speech_eeg.model import feature_transform,forecast
        from speech_eeg.cohort_data import design
        import torch
        torch.set_num_threads(1)
        support=validate_features(features);ids=np.flatnonzero(support)
        y=np.full((len(support),40),np.nan)
        if len(ids):
            x,p=feature_transform(features,self.state,ids)
            anchor=design(x,p,self.slots['acoustic'])@self.ridge['coef']+self.ridge['intercept']
            neural=np.mean([forecast(m,x[:,self.slots['full']],p) for m in self.models],axis=0)
            y[ids]=((1-self.weight)*anchor+self.weight*neural)*self.state['target_scale']+self.state['target_mean']
            if not np.isfinite(y[ids]).all():raise ValueError('nonfinite supported EEG prediction')
        return y,support.copy(),self.weight

def pcm44100(audio, output):
    """Canonical speech adapter PCM; byte-preserve already matching originals."""
    audio=Path(audio); out=Path(output)
    try:
        with wave.open(str(audio)) as f:
            if (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getcomptype())==(1,2,44100,'NONE'):
                return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy(),None
    except wave.Error:pass
    guard(out)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(audio),'-vn','-ac','1','-ar','44100','-c:a','pcm_s16le',str(out)],check=True,timeout=1800)
    with wave.open(str(out)) as f:pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
    return pcm,{'file':out.name,'sha256':file_sha256(out),'policy':'FFmpeg mono PCM16 44100Hz; no stretch; source begins at decoded sample zero'}

def acoustic_points(pcm):
    """Exact training RMS arithmetic, without an entire-file int64 prefix."""
    n=len(pcm);q=np.arange(0,(n*500+44099)//44100,10,dtype=np.int64)
    starts=((q-5)*44100+499)//500;stops=((q+5)*44100+499)//500
    valid=(starts>=0)&(stops<=n)&(stops>starts)
    rms=np.full(len(q),np.nan)
    for i in np.flatnonzero(valid):
        x=pcm[starts[i]:stops[i]].astype(np.int64)
        rms[i]=np.sqrt(float(x@x)/((stops[i]-starts[i])*32768.**2))
    onset=np.full(len(q),np.nan);ix=np.flatnonzero(valid[1:]&valid[:-1])+1
    onset[ix]=np.maximum(0,rms[ix]-rms[ix-1])
    p=q.astype(np.float64)*44100/(500*n)
    return q/500,np.c_[rms,onset],np.c_[p,p**2,p**3]

def speech_projected(pcm, source_sha, output):
    """Frozen exact CPU/chunk/resampling policy; output only training-PCA states."""
    from av_eeg_encoding.features.alice_model_audio import pcm16_to_model_audio
    from av_eeg_encoding.features.narrative_temporal_audio import FrozenNarrativeTemporalWav2Vec2Encoder,plan_chunks
    import torch
    torch.set_num_threads(1)
    if file_sha256(PCA)!=PCA_SHA:raise ValueError('training PCA changed')
    pca=load(PCA);guard(output)
    model_audio=pcm16_to_model_audio(pcm,guard=lambda _:guard(output))
    encoder=FrozenNarrativeTemporalWav2Vec2Encoder(SNAPSHOT,expected_receipt_sha256=RECEIPT,
        expected_versions={'torch':'2.14.1','transformers':'5.20.2'},guard=lambda _:guard(output))
    projected=[];starts=[];began=time.monotonic()
    for c in plan_chunks(len(model_audio.values)):
        guard(output)
        rows=encoder.encode_chunk(model_audio.values[c.start_sample:c.stop_sample],chunk=c,
            source_sha256=source_sha,source_sample_count=len(model_audio.values),
            source_resampler_support_seconds=(0.,len(pcm)/44100))
        states=rows.values
        projected.append((states.astype(np.float64)-pca['mean'])@pca['axes'])
        starts.append(c.start_sample+np.arange(c.frame_count)*320)
        print(json.dumps({'stage':'wav2vec','chunk':c.index,'frames':c.frame_count,'elapsed':time.monotonic()-began}),flush=True)
    result={'values':np.concatenate(projected),'ticks':np.concatenate(starts)*4+798}
    np.savez(Path(output)/'speech-projected.npz',**result)
    return result

def native_points(native_root):
    """Flatten chronological native 2 Hz batches without inventing finer support."""
    native_root=Path(native_root);m=read(native_root/'manifest.json')
    qualified = m.get('hidden_capture_native_outputs_bitwise_equal') or m.get('replay_qualification',{}).get('passed')
    if not qualified or not m.get('hidden_features'):
        raise ValueError('qualified native audio hidden representations required')
    if m.get('input_modalities')!=['text','audio'] and set(m.get('input_modalities',[]))!={'text','audio'}:
        raise ValueError('speech EEG requires audio/text-only TRIBE input')
    identity=read(ROOT/'artifacts/cohort-v1/manifest.json')['native_encoder_identity']
    if (not m.get('content_encoder_revisions_verified') or any(m.get(k)!=identity[k] for k in
            ('checkpoint_revision','checkpoint_sha256','upstream_config_sha256','observed_encoder_revisions','torch_version'))):
        raise ValueError('native audio checkpoint or encoder identity differs from waveform training')
    pieces=[]
    for h,d in zip(m['hidden_features'],m['dense_features'],strict=True):
        if h['frequency_hz']!=2 or d['frequency_hz']!=2 or h['segments']!=[{k:s[k] for k in ('start','duration')} for s in d['segments']]:
            raise ValueError('native capture clocks differ')
        for r in(h,d):
            if file_sha256(native_root/r['file'])!=r['sha256']:raise ValueError('native array checksum changed')
        hs=load(native_root/h['file'])['hidden'];ds=load(native_root/d['file'])
        for i,s in enumerate(h['segments']):
            values={'tribe_hidden':hs[i],**{f'native_{k}':ds[k][i].reshape(-1,ds[k].shape[-1]).T for k in ('audio','text')}}
            if any(len(v)!=round(s['duration']*2) for v in values.values()):raise ValueError('native sample clock mismatch')
            pieces.append((s['start'],values))
    by_tick={}
    # Prefer later-starting context if native chunks overlap; fixed before scoring.
    for start,values in sorted(pieces,key=lambda p:p[0]):
        for i in range(len(values['tribe_hidden'])):
            tick=round((start+i/2)*500)
            if abs(tick/500-(start+i/2))>1e-8:raise ValueError('nonintegral native tick')
            by_tick[tick]={k:v[i] for k,v in values.items()}
    ticks=np.array(sorted(by_tick),dtype=np.int64)
    return {k:np.stack([by_tick[t][k] for t in ticks]) for k in ('native_audio','native_text','tribe_hidden')},ticks,m

def base_history(query, acoustic, speech, ids):
    ticks=np.rint(query[ids]*500).astype(np.int64)[:,None]-np.arange(50,-1,-1)[None,:]*10
    ix=ticks//10;v=(ix>=0)&(ix<len(query));ac=acoustic[np.clip(ix,0,len(query)-1)]
    req=ticks*128;ni=np.searchsorted(speech['ticks'],req,side='right')-1
    ok=(ni>=0);age=req-speech['ticks'][np.clip(ni,0,len(speech['ticks'])-1)]
    ok &= age<=1280
    sp=speech['values'][np.clip(ni,0,len(speech['values'])-1)]
    values=np.concatenate([ac,sp],axis=2).transpose(0,2,1).astype(np.float32)
    support=(v&ok&np.isfinite(ac).all(2)).all(1)
    values[np.broadcast_to(~(v&ok)[:,None,:],values.shape)]=np.nan
    return values,support,ticks

def feature_batches(query, acoustic,progress,speech,native,nticks,artifact,batch_size=512,native_offset_seconds=0.):
    states={b:load(Path(artifact)/(b+'-pca.npz')) for b in native}
    projected={b:project(native[b],states[b]) for b in native}
    for start in range(0,len(query),batch_size):
        ids=np.arange(start,min(start+batch_size,len(query)));base,support,ticks=base_history(query,acoustic,speech,ids)
        native_requests=ticks if native_offset_seconds==0 else ticks.astype(float)+float(native_offset_seconds)*500
        ii=np.searchsorted(nticks,native_requests,side='right')-1
        ok=(ii>=0)&((native_requests-nticks[np.clip(ii,0,len(nticks)-1)])<250)
        histories=[base]
        for b in ('native_audio','native_text','tribe_hidden'):
            h=projected[b][np.clip(ii,0,len(nticks)-1)].transpose(0,2,1).astype(np.float32)
            h[np.broadcast_to(~ok[:,None,:],h.shape)]=np.nan;histories.append(h)
        yield ids,{'history':np.concatenate(histories,axis=1),'progress':progress[ids],
                   'query_seconds':query[ids],'feature_supported':support&ok.all(1)}

def preflight(endpoint_run=ROOT/'runs/endpoint-v1'):
    run=Path(endpoint_run);m=read(run/'development-selection.json')
    if m['full']['weight']!=.2:raise ValueError('fixed endpoint blend changed')
    for f in('forecast-lock.json','config.json','transforms.npz'):
        if not(run/f).is_file():raise ValueError('missing endpoint artifact')
    artifacts=ROOT/'artifacts/cohort-v1'
    bundle=read(artifacts/'manifest.json')
    for name,digest in bundle['files'].items():
        member=(artifacts/name).resolve()
        if not member.is_relative_to(artifacts.resolve()) or file_sha256(member)!=digest:
            raise ValueError('waveform input-transform artifact changed')
    return {'endpoint_run':str(run),'forecast_lock_sha256':file_sha256(run/'forecast-lock.json'),'channels':read(ROOT/'runs/cohort-v1/started.json')['channels']}

def predict(audio_path,audio_native_root,output,*,source_sha256=None,source_offset_seconds=0.,endpoint_run=ROOT/'runs/endpoint-v1',parity_receipt=None,audio_lineage=None,speech_cache=None):
    if importlib.util.find_spec('torch') is None:
        from brain.fusion_inference import torch_python
        args=[torch_python(),'-B','-m','numerical_brain_v1.waveform',str(Path(audio_path).resolve()),
              str(Path(audio_native_root).resolve()),str(Path(output).resolve()),'--endpoint-run',str(Path(endpoint_run).resolve()),
              '--source-offset-seconds',str(source_offset_seconds)]
        for flag,value in [('source-sha256',source_sha256),('parity-receipt',parity_receipt),('audio-lineage',audio_lineage),('speech-cache',speech_cache)]:
            if value is not None:args+=['--'+flag,str(Path(value).resolve()) if flag!='source-sha256' else value]
        subprocess.run(args,cwd=ROOT,check=True,timeout=3600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'))
        return read(Path(output)/'manifest.json')
    audio_path,audio_native_root,output=map(Path,(audio_path,audio_native_root,output));identity=preflight(endpoint_run)
    parity_receipt=Path(parity_receipt or ROOT/'.local/numerical-brain-v1/waveform-parity-r3/report.json')
    if not read(parity_receipt).get('passed'):raise ValueError('real speech feature parity required first')
    guard(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic();audio_sha=file_sha256(audio_path)
    native,nticks,nm=native_points(audio_native_root)
    lineage=None
    if nm['source']['sha256']!=audio_sha:
        path=Path(audio_lineage) if audio_lineage else audio_path.parent/'manifest.json'
        lineage=read(path)
        if (lineage.get('schema')!='perception-audio-derivative-v1'
                or lineage['parent']['sha256']!=nm['source']['sha256']
                or lineage['parent']['sha256']!=source_sha256
                or lineage['derivative']['sha256']!=audio_sha
                or not lineage.get('audio_clock_continuity_checked')
                or abs(lineage['first_native_audio_seconds']-source_offset_seconds)>1e-8):
            raise ValueError('audio and native features lack a verified derivative clock/identity')
    pcm,derivative=pcm44100(audio_path,output/'canonical.wav');duration=len(pcm)/44100
    query,acoustic,progress=acoustic_points(pcm)
    if speech_cache is None:
        speech=speech_projected(pcm,audio_sha,output)
    else:
        cached=Path(speech_cache);receipt=read(cached/'manifest.json')
        if (receipt['audio_source_sha256']!=audio_sha or receipt['pcm_frames']!=len(pcm)
                or receipt['pca_sha256']!=PCA_SHA
                or file_sha256(cached/'speech-projected.npz')!=receipt['files']['speech-projected.npz']):
            raise ValueError('cached speech source or PCA mismatch')
        speech=load(cached/'speech-projected.npz')
    feature_time=time.monotonic()-started
    y=np.full((len(query),40),np.nan);support=np.zeros(len(query),bool)
    predictor=EndpointPredictor(endpoint_run)
    native_offset=source_offset_seconds if lineage is not None else 0.
    for ids,features in feature_batches(query,acoustic,progress,speech,native,nticks,ROOT/'artifacts/cohort-v1',native_offset_seconds=native_offset):
        guard(output);pred,ok,w=predictor.predict(features)
        ok&=(query[ids]>=2)&(query[ids]+2<math.ceil(duration*500)/500)
        y[ids[ok]]=pred[ok];support[ids]=ok
    q=query+float(source_offset_seconds)
    np.savez(output/'eeg.npz',predicted_eeg_volts=y,query_seconds=q,supported=support)
    m={'schema':'perception-numerical-waveform-v1','complete':True,'source_sha256':source_sha256 or audio_sha,
       'audio_source_sha256':audio_sha,'audio_native_manifest_sha256':file_sha256(audio_native_root/'manifest.json'),
       **identity,'channels':identity['channels'],'unit':'V','rate_hz':50,'filter_hz':[.5,8.],
       'target':'filtered average-listener EEG; not full-band raw or individual EEG','neural_blend_weight':.2,
       'source_offset_seconds':float(source_offset_seconds),'duration_seconds':duration,'supported_samples':int(support.sum()),
       'audio_lineage':lineage,
       'native_feature_clock_offset_seconds':float(native_offset),
       'total_samples':len(q),'experimental_transfer':True,'canonical_audio':derivative,
       'parity_receipt_sha256':file_sha256(parity_receipt),'inference_seconds':time.monotonic()-started-feature_time,
       'preprocessing_seconds':feature_time,'foundation_TRIBE_inference_included':False,
       'speech_feature_cache_reused':speech_cache is not None,
       'speech_feature_cache_manifest_sha256':file_sha256(Path(speech_cache)/'manifest.json') if speech_cache is not None else None,
       'speech_feature_original_preparation_seconds':receipt.get('elapsed_seconds') if speech_cache is not None else None,
       'timing_scope':'Local preprocessing/cache reads and loaded waveform prediction; native TRIBE extraction excluded. Cached speech preparation is reported separately, not charged to this run.',
       'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}}
    write(output/'manifest.json',m);return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('native');p.add_argument('output');p.add_argument('--source-sha256')
    p.add_argument('--source-offset-seconds',type=float,default=0.);p.add_argument('--endpoint-run',default=str(ROOT/'runs/endpoint-v1'))
    p.add_argument('--parity-receipt');p.add_argument('--audio-lineage');p.add_argument('--speech-cache')
    a=p.parse_args();print(json.dumps(predict(a.audio,a.native,a.output,source_sha256=a.source_sha256,source_offset_seconds=a.source_offset_seconds,
        endpoint_run=a.endpoint_run,parity_receipt=a.parity_receipt,audio_lineage=a.audio_lineage,speech_cache=a.speech_cache),indent=2))
