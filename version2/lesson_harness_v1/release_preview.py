"""Self-contained scientific preview of exported predictions, no network or UI edits."""
import argparse
import json
from pathlib import Path

import numpy as np

from brain.edugage_targets import guard
from brain.worker_contract import file_sha256


def build(response_path, output):
    response_path,output=Path(response_path),Path(output)
    guard(output)
    if output.exists():
        raise ValueError('preserve previous preview')
    response=json.loads(response_path.read_text())
    if response['schema']!='perception-browser-brain-response-v1':
        raise ValueError('verified browser response required')
    cortical=response['fmri'];root=response_path.parent.resolve()
    path=(root/cortical['file']).resolve()
    if not path.is_relative_to(root) or file_sha256(path)!=cortical['sha256']:
        raise ValueError('cortical buffer identity differs')
    dtype=np.dtype(cortical['dtype']);shape=tuple(cortical['shape'])
    if (dtype.kind!='f' or dtype.itemsize not in (4,8) or len(shape)!=2
            or path.stat().st_size!=int(np.prod(shape))*dtype.itemsize):
        raise ValueError('invalid cortical buffer')
    values=np.memmap(path,mode='r',dtype=dtype,shape=shape)
    rms=[float(np.sqrt(np.mean(np.square(row.astype(np.float64))))) for row in values]
    if not np.isfinite(rms).all():
        raise ValueError('nonfinite native summary')
    del values
    data={'response':response,'native_rms':rms,'response_sha256':file_sha256(response_path)}
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    html=TEMPLATE.replace('__DATA__',payload)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html)
    report={'complete':True,'self_contained':True,'remote_requests':0,
        'response_sha256':file_sha256(response_path),'html_sha256':file_sha256(output),
        'fmri_summary':'root mean square across all cortical vertices per unchanged native frame',
        'native_array_modified':False,'anatomical_mapping_used':False,
        'student_accuracy_claimed':False}
    output.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


TEMPLATE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lesson model — scientific preview</title>
<style>
:root{color-scheme:light;font:15px system-ui,sans-serif;color:#1d2735;background:#f5f6f8}*{box-sizing:border-box}body{max-width:1160px;margin:0 auto;padding:36px 24px}header{margin-bottom:26px}h1{font-size:30px;letter-spacing:-1px;margin:9px 0}h2{font-size:16px;margin:0}p{line-height:1.55}.muted{color:#616c7b}.badge{display:inline-block;background:#e8edf4;border-radius:5px;padding:5px 9px;font-size:12px;margin-right:6px}.notice{border-left:3px solid #b28639;padding:10px 14px;background:#fff8ec;font-size:14px}.controls{position:sticky;top:0;background:#f5f6f8eF;backdrop-filter:blur(8px);padding:12px 0;z-index:2;display:flex;align-items:center;gap:16px;border-bottom:1px solid #dce1e7}input{flex:1;accent-color:#345b94}output{font:600 16px ui-monospace,monospace;min-width:80px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:20px 0}.card{padding:18px;background:white;border:1px solid #dfe4eb;border-radius:8px}.card-head{display:flex;align-items:start;justify-content:space-between;gap:12px}.score{font-size:26px;line-height:1;font-variant-numeric:tabular-nums}.state{font-size:12px;min-height:22px;margin:8px 0;color:#536581}.chart{width:100%;height:135px}.axis{font-size:10px;fill:#798392}.description{font-size:12px;margin:4px 0 0;min-height:34px;color:#6b7480}.wide{grid-column:1/-1}.native .chart{height:160px}.readout{font-size:12px;line-height:1.7}.key{display:flex;gap:18px;font-size:12px;color:#687488}details{margin:14px 0;padding:14px;border:1px solid #dfe4eb;background:#fff;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px ui-monospace,monospace}.value-null{color:#929aa6}.footer{font-size:12px;color:#687488;padding:12px 0 30px}@media(max-width:700px){.grid{grid-template-columns:1fr}body{padding:20px 14px}h1{font-size:25px}.controls{gap:8px}}
</style>
<header><span class="badge">LOCAL EXPERIMENTAL MODEL</span><span class="badge">SCIENTIFIC PREVIEW</span><h1>Lesson predictions, over time</h1><p class="muted" id="subtitle"></p><p class="notice">These are model predictions for an assumed reference learner. Synthetic quality labels are content-based estimates, not measurements of a student's attention or understanding. The separate TRIBE stream predicts cortical responses; EEG training has not modified TRIBE.</p></header>
<div class="controls"><label for="time">Lesson time</label><input id="time" type="range" min="0" step="1"><output id="clock"></output></div>
<div class="grid" id="cards"></div>
<section class="card native"><div class="card-head"><h2>Native TRIBE cortical response</h2><span class="badge">UNCHANGED FROZEN BRANCH</span></div><p class="description">Response magnitude (RMS across all cortical vertices) at each native frame. This is a compact numerical summary, not an anatomical map, attention score, or measured fMRI.</p><div id="native-chart"></div><p class="readout" id="native-detail"></p></section>
<details><summary>Prediction scope and provenance</summary><div id="scope" class="readout"></div><pre id="identity"></pre></details>
<p class="footer">Move the slider to inspect the exact available query and native frame. Quality positions use the preceding 44 seconds; adjacent positions heavily overlap. Unsupported values stay unavailable. Categories summarize each predicted ordinal distribution and are not separately validated classifiers.</p>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent),R=DATA.response;
const names={attention_difficulty:'Attention difficulty',interest_engagement:'Interest / engagement',mental_effort:'Mental effort',confusion_risk:'Confusion risk',comprehension_difficulty:'Comprehension difficulty'};
const el=id=>document.getElementById(id),duration=R.duration_seconds,nativeDuration=Math.max(duration,...R.fmri.segments.map(s=>s.start+s.duration));
el('subtitle').textContent=`${R.input_format.toUpperCase()} · ${duration.toFixed(1)} seconds · 1-second quality queries · 44-second context`;
el('time').max=Math.floor(duration);el('time').value=Math.min(44,Math.floor(duration));
function chart(points,maxY,id,color){
 const span=id==='native'?nativeDuration:duration,w=520,h=140,l=30,t=10,b=22,r=10,x=s=>l+Math.max(0,s)/span*(w-l-r),y=v=>h-b-v/maxY*(h-b-t);
 let d='',open=false;for(const [s,v]of points){if(v===null){open=false;continue}d+=(open?'L':'M')+x(s).toFixed(2)+','+y(v).toFixed(2)+' ';open=true}
 return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted values over lesson time"><rect x="${l}" y="${t}" width="${id==='native'?0:Math.max(0,x(Math.min(44,duration))-l)}" height="${h-b-t}" fill="#f4f5f7"/><line x1="${l}" y1="${h-b}" x2="${w-r}" y2="${h-b}" stroke="#d9dfe7"/><line x1="${l}" y1="${y(maxY/2)}" x2="${w-r}" y2="${y(maxY/2)}" stroke="#edf0f4"/><text x="2" y="${t+7}" class="axis">${maxY.toFixed(maxY===10?0:2)}</text><text x="14" y="${h-b}" class="axis">0</text><text x="${l}" y="${h-5}" class="axis">0s</text><text x="${w-r}" y="${h-5}" text-anchor="end" class="axis">${Math.floor(span)}s</text><path d="${d}" fill="none" stroke="${color}" stroke-width="2.2"/><line id="cursor-${id}" y1="${t}" y2="${h-b}" stroke="#b9c3d2" stroke-dasharray="3 3"/></svg>`;
}
for(const q of Object.keys(names)){
 const card=document.createElement('article');card.className='card';
 card.innerHTML=`<div class="card-head"><h2>${names[q]}</h2><strong class="score" id="score-${q}"></strong></div><p class="state" id="state-${q}"></p>${chart(R.timeline.map(p=>[p.second,p.qualities[q].value]),10,q,'#456b9b')}<p class="description" id="description-${q}"></p>`;
 el('cards').appendChild(card);el(`description-${q}`).textContent=R.definitions[q];
}
const nativeMax=Math.max(...DATA.native_rms)*1.08||1;
el('native-chart').innerHTML=chart(R.fmri.segments.map((s,i)=>[s.start,DATA.native_rms[i]]),nativeMax,'native','#7e6a9e');
function update(){
 const second=Number(el('time').value),point=R.timeline[second];el('clock').textContent=`${second}s`;
 const cursorX=30+second/duration*480;
 for(const q of Object.keys(names)){
  const v=point.qualities[q],score=el(`score-${q}`);score.textContent=v.value===null?'—':v.value.toFixed(2);score.classList.toggle('value-null',v.value===null);
  const origins=v.supervision_origins.map(s=>s==='human'?'human labels':'AI content review').join(' + ');
  const kind=v.evidence_status==='human_supported'?`Exploratory · ${origins}`:v.evidence_status==='synthetic_proxy'?'Synthetic AI content-review proxy':v.evidence_status==='experimental_transfer'?`Experimental transfer · ${origins}`:'Unavailable';
  el(`state-${q}`).textContent=v.value===null?`${kind} · ${v.reason.replaceAll('_',' ')}`:`${v.category.id.replaceAll('_',' ')} · ${kind} · /10`;
  el(`cursor-${q}`).setAttribute('x1',cursorX);el(`cursor-${q}`).setAttribute('x2',cursorX);
 }
 const nativeX=30+second/nativeDuration*480;el('cursor-native').setAttribute('x1',nativeX);el('cursor-native').setAttribute('x2',nativeX);
 const i=R.fmri.segments.findIndex(s=>second>=s.start&&second<s.start+s.duration);
 el('native-detail').textContent=i<0?'No native frame covers this selected time.':`Native frame ${i} · interval ${R.fmri.segments[i].start.toFixed(6)}–${(R.fmri.segments[i].start+R.fmri.segments[i].duration).toFixed(6)}s · RMS ${DATA.native_rms[i].toPrecision(5)} model-response units · ${R.fmri.shape[1].toLocaleString()} cortical values in this frame`;
}
el('time').addEventListener('input',update);update();
el('scope').textContent=`Reference population: ${R.reference_population}. Native array shape: ${R.fmri.shape.join(' × ')}. Spatial vertex ordering qualified: ${R.fmri.spatial_vertex_order_qualified}. No named anatomical regions are inferred. Full native arrays remain in the accompanying cortical-response.bin and fmri.npz files. No participant recordings or model weights are embedded in this preview.`;
el('identity').textContent=`Model: ${R.model_version}\nArtifact SHA-256: ${R.artifact_sha256}\nSource SHA-256: ${R.source_sha256}\nResponse SHA-256: ${DATA.response_sha256}\nTRIBE array SHA-256: ${R.fmri.sha256}\nBehavioral accuracy: see accompanying benchmark.json; this display is not a validation result.`;
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('response',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(build(args.response,args.output),indent=2))
