"""Prepare only training/development waveform targets under fixed metadata policy."""
import argparse
from pathlib import Path
import wave
import numpy as np
from scipy.signal import firwin,fftconvolve,freqz
from .common import read,write,sha,guard


def kernel(config):
    f=config['filter'];h=firwin(f['numtaps'],f['cutoff_hz'],window=f['window'],fs=f['fs'],pass_zero=False,scale=True)
    return h-h.mean()


def filtered_target(raw,h):
    """Raw channel x samples -> full phase-zero grid x channels, NaN off support."""
    n=raw.shape[1];q=np.arange(0,n,10);half=(len(h)-1)//2
    y=np.full((len(q),len(raw)),np.nan);mask=np.zeros_like(y,bool)
    geometry=(q>=half)&(q+half<n);ix=np.flatnonzero(geometry)
    if not len(ix):return y,mask,q
    for ch,x in enumerate(raw):
        finite=np.isfinite(x)
        # Zero here is numerical convolution scratch only. Missing-context outputs stay NaN.
        v=fftconvolve(np.where(finite,x,0.),h,mode='valid')
        missing=np.r_[0,np.cumsum(~finite)]
        supported=(missing[q[ix]+half+1]-missing[q[ix]-half])==0
        rows=ix[supported];y[rows,ch]=v[q[rows]-half];mask[rows,ch]=True
    return y,mask,q


def run(config_path,output):
    import mne
    c=read(config_path);resource=read('configs/stage_a.json');guard(resource)
    audit=read(c['header_audit']);channels=audit['training_common_good_channels']
    assert len(channels)==40
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    h=kernel(c);np.savez(output/'filter.npz',coefficients=h)
    freqs=np.array([0.,.25,.5,1.,2.,4.,8.,12.,25.]);_,response=freqz(h,worN=freqs,fs=500)
    records=[]
    sources=Path('/Volumes/EEG Research Workspace/raw/alice-s01-format-v1-2026-10-27/stimuli')
    for r in audit['participants']:
        person=r['participant']
        if person in c['reserved_test_people']:continue
        segments=c['training_segments'] if person in c['training_people'] else c['development_segments']
        p=Path(r['path']);assert sha(p)==r['sha256'];guard(resource)
        raw=mne.io.read_raw_fif(p,preload=False,verbose='ERROR')
        if raw.info['sfreq']!=500 or raw.first_samp!=0:raise ValueError('Unexpected native EEG clock')
        picks=[raw.ch_names.index(ch) for ch in channels]
        if any(raw.get_channel_types()[i]!='eeg' or raw.info['chs'][i]['unit']!=107 for i in picks):
            raise ValueError('Expected EEG in FIF-declared volts')
        good=np.array([ch not in r['bads'] for ch in channels])
        for segment in segments:
            wav=sources/f'{segment}.wav'
            with wave.open(str(wav)) as w:
                frames,rate=w.getnframes(),w.getframerate()
            n=(frames*500+rate-1)//rate;start=r['event_samples'][str(segment)];stop=start+n
            if stop>raw.n_times:raise ValueError('Audio exceeds recorded EEG extent')
            next_start=r['event_samples'].get(str(segment+1),raw.n_times)
            if stop>next_start:raise ValueError('Source segments overlap')
            # Header-flagged channels are retained as explicitly unsupported columns.
            values=raw.get_data(picks=picks,start=start,stop=stop);values[~good]=np.nan
            target,support,q=filtered_target(values,h)
            file=output/f'{person}-segment{segment:02d}.npz'
            np.savez(file,target=target,supported=support,query_seconds=q/500,
                query_native_indices=q,source_global_samples=start+q,good_channels=good)
            records.append({'participant':person,'segment':segment,'file':file.name,'sha256':sha(file),
                'source_sha256':r['sha256'],'audio_sha256':sha(wav),'source_start':start,'source_stop':stop,
                'supported_rows_by_channel':support.sum(0).tolist()})
        raw.close();print(person+' training/development targets prepared',flush=True)
    for segment in c['training_segments']:
        items=[r for r in records if r['segment']==segment and r['participant'] in c['training_people']]
        assert len(items)==4;ys=[];ms=[];clock=None
        for r in items:
            with np.load(output/r['file'],allow_pickle=False) as z:
                if clock is None:clock=z['query_seconds']
                np.testing.assert_array_equal(clock,z['query_seconds']);ys.append(z['target']);ms.append(z['supported'])
        mask=np.logical_and.reduce(ms);mean=np.mean(ys,axis=0);mean[~mask]=np.nan
        file=output/f'train-group-segment{segment:02d}.npz';np.savez(file,target=mean,supported=mask,query_seconds=clock)
        records.append({'participant':'training_group','segment':segment,'file':file.name,'sha256':sha(file)})
    write(output/'manifest.json',{'schema':'perception-v2-cohort-targets','complete':True,'config':c,
        'config_sha256':sha(config_path),'header_audit_sha256':sha(c['header_audit']),'code_sha256':sha(__file__),
        'channels':channels,'files':records,'reserved_test_EEG_decoded':False,
        'filter_response':{'frequency_hz':freqs.tolist(),'gain':np.abs(response).tolist(),'DC_sum':float(h.sum()),'future_context_seconds':2.0},
        'target_units':'FIF-declared volts; source assertion, not independent physical calibration'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args();run(a.config,a.output)
