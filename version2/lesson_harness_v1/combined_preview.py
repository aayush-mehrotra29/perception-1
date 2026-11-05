"""Self-contained playback of three precomputed prediction streams; no fake EEG."""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256
from .app_release import checked_child
from .combined_export import validate_eeg
from .release_preview import TEMPLATE


EEG_PANEL='''<section class="card" style="margin-bottom:20px"><div class="card-head"><h2>Predicted EEG summaries · AF7 / AF8</h2><span class="badge">44-SECOND WINDOWS</span></div><p class="description">Channel means and log1p variances from the saved EEG readout. These are coarse summary predictions, not oscillating raw EEG waveforms, frequency bands, or measured student activity. Received voltage offsets remain in the mean targets.</p><div class="grid"><div><h2>Mean voltage · μV</h2><div id="eeg-means"></div></div><div><h2>Log1p variance · log1p(μV²)</h2><div id="eeg-variances"></div></div></div><div class="key"><span style="color:#456b9b">● AF7</span><span style="color:#a57936">● AF8</span></div><p id="eeg-detail" class="readout"></p></section>'''

EEG_SCRIPT='''
const EEG=DATA.eeg;
function eegChart(keys,id){
 const vals=EEG.points.flatMap(p=>keys.map(k=>p.values[k])).filter(v=>v!==null);
 const a=vals.length?Math.min(...vals):0,b=vals.length?Math.max(...vals):1,pad=Math.max((b-a)*.12,1e-4),lo=a-pad,hi=b+pad;
 const w=520,h=140,l=50,r=10,t=10,bt=22,x=s=>l+s/duration*(w-l-r),y=v=>h-bt-(v-lo)/(hi-lo)*(h-bt-t);
 let paths='';keys.forEach((key,j)=>{let d='',open=false;for(const p of EEG.points){const v=p.values[key];if(v===null){open=false;continue}d+=(open?'L':'M')+x(p.second).toFixed(2)+','+y(v).toFixed(2)+' ';open=true}paths+=`<path d="${d}" fill="none" stroke="${j?'#a57936':'#456b9b'}" stroke-width="2"/>`});
 return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted EEG window summaries over time"><rect x="${l}" y="${t}" width="${Math.max(0,x(Math.min(44,duration))-l)}" height="${h-bt-t}" fill="#f4f5f7"/><line x1="${l}" y1="${h-bt}" x2="${w-r}" y2="${h-bt}" stroke="#d9dfe7"/><text x="2" y="${t+7}" class="axis">${hi.toPrecision(4)}</text><text x="2" y="${h-bt}" class="axis">${lo.toPrecision(4)}</text><text x="${l}" y="${h-5}" class="axis">0s</text><text x="${w-r}" y="${h-5}" text-anchor="end" class="axis">${Math.floor(duration)}s</text>${paths}<line id="cursor-${id}" y1="${t}" y2="${h-bt}" stroke="#b9c3d2" stroke-dasharray="3 3"/></svg>`;
}
el('eeg-means').innerHTML=eegChart(['mean_AF7','mean_AF8'],'eeg-mean');
el('eeg-variances').innerHTML=eegChart(['log1p_variance_AF7','log1p_variance_AF8'],'eeg-variance');
function updateEEG(){
 const second=Number(el('time').value),p=EEG.points[second],x=50+second/duration*460;
 for(const id of ['eeg-mean','eeg-variance']){el(`cursor-${id}`).setAttribute('x1',x);el(`cursor-${id}`).setAttribute('x2',x)}
 el('eeg-detail').textContent=p.status==='unavailable'?`Unavailable at ${second}s · ${p.reason.replaceAll('_',' ')}`:`Context ${p.context_start}–${p.context_stop}s · AF7 mean ${p.values.mean_AF7.toFixed(3)} μV, log1p variance ${p.values.log1p_variance_AF7.toFixed(3)} · AF8 mean ${p.values.mean_AF8.toFixed(3)} μV, log1p variance ${p.values.log1p_variance_AF8.toFixed(3)} · ${p.status.replaceAll('_',' ')}`;
}
'''

PLAYBACK_SCRIPT='''
let playing=false,playPosition=Number(el('time').value),lastTick=null;
el('play').addEventListener('click',()=>{if(Number(el('time').value)>=Math.floor(duration)){el('time').value=0;update()}playPosition=Number(el('time').value);playing=!playing;lastTick=null;el('play').textContent=playing?'Pause':'Play'});
el('time').addEventListener('input',()=>{playing=false;playPosition=Number(el('time').value);lastTick=null;el('play').textContent='Play'});
document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;lastTick=null;el('play').textContent='Play'}});
function playbackFrame(stamp){if(playing){if(lastTick!==null){playPosition+=Math.max(0,stamp-lastTick)/1000*Number(el('speed').value);el('time').value=Math.min(Math.floor(duration),Math.floor(playPosition));update();if(playPosition>=Math.floor(duration)){playing=false;el('play').textContent='Replay'}}lastTick=stamp}requestAnimationFrame(playbackFrame)}
requestAnimationFrame(playbackFrame);
'''


def build(source,output):
    source,output=Path(source),Path(output);guard(output)
    if output.exists():raise ValueError('preserve previous playback preview')
    root=source.parent
    receipt=json.loads((root/'combined-verification.json').read_text())
    if not receipt.get('complete'):raise ValueError('verified three-stream package required')
    for name,digest in receipt['files'].items():checked_child(root,name,digest)
    combined=json.loads(source.read_text());response=combined['quality_response']
    if (combined['schema']!='perception-three-stream-response-v1' or combined['eeg_waveform_generated']
            or combined['qualities_consume_predicted_eeg_values']):
        raise ValueError('explicit parallel summary architecture required')
    eeg=validate_eeg(combined['eeg'],response)
    cortical=response['fmri'];path=checked_child(root,cortical['file'],cortical['sha256'])
    values=np.memmap(path,mode='r',dtype=np.dtype(cortical['dtype']),shape=tuple(cortical['shape']))
    rms=[float(np.sqrt(np.mean(np.square(row.astype(np.float64))))) for row in values];del values
    if not np.isfinite(rms).all():raise ValueError('nonfinite cortical summary')
    data={'response':response,'eeg':eeg,'native_rms':rms,'response_sha256':file_sha256(source)}
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    html=TEMPLATE.replace('__DATA__',payload)
    html=html.replace('Lesson model — scientific preview','Three-stream lesson model — offline playback')
    html=html.replace('Lesson predictions, over time','EEG, cortical and lesson-quality predictions')
    html=html.replace('<label for="time">Lesson time</label>', '<button id="play" style="padding:8px 14px;border:1px solid #bfc9d7;background:white;border-radius:5px">Play</button><label for="speed">Speed</label><select id="speed" style="padding:7px"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><label for="time">Time</label>')
    html=html.replace('<section class="card native">',EEG_PANEL+'<section class="card native">')
    html=html.replace('function update(){',EEG_SCRIPT+'\nfunction update(){\n updateEEG();')
    html=html.replace('</script></html>',PLAYBACK_SCRIPT+'</script></html>')
    html=html.replace('Move the slider to inspect the exact available query and native frame.',
        'Play, pause or scrub the precomputed streams. No new inference runs during playback; this is not a validated causal real-time brain forecast. Category scores and EEG summaries are parallel outputs of the shared trained representation, rather than categories computed from the displayed EEG numbers.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html)
    report={'complete':True,'source_sha256':file_sha256(source),'html_sha256':file_sha256(output),
        'self_contained':True,'remote_requests':0,'three_streams':['quality_ratings','EEG_window_summaries','native_TRIBE_cortical_response'],
        'raw_waveform_synthesis':False,'causal_realtime_claimed':False,'original_predictions_modified':False}
    output.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(build(a.source,a.output),indent=2))
