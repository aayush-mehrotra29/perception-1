"""Local narrated-presentation adapter; actual audio determines generated clocks."""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave

from brain.edugage_targets import guard
from brain.worker_contract import PublicMediaRequest,file_sha256,validate_media

RATE=48000
FPS=25


def command(args,timeout=300):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout).stdout


def probe(source):
    return json.loads(command(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(source)]))


def caption(records,index,source_hash):
    matches=[r for r in (records or []) if r.get('source_page_index')==index]
    if len(matches)!=1:raise ValueError('exactly one source-bound descriptive caption required for non-text page')
    row=matches[0]
    if (row.get('source_sha256')!=source_hash or row.get('kind')!='generated_descriptive_caption'
            or not row.get('model') or not re.fullmatch('[0-9a-f]{64}',row.get('provenance_sha256',''))
            or not isinstance(row.get('text'),str) or not row['text'].strip()):
        raise ValueError('qualified descriptive caption provenance required')
    return row['text'],dict(row)


def clock(samples,*,video):
    if type(samples) is not int or samples<1:raise ValueError('nonempty audio required')
    display_samples=math.ceil(samples/(RATE//FPS))*(RATE//FPS) if video else samples
    return {'spoken_samples':samples,'display_samples':display_samples,
        'spoken_seconds':samples/RATE,'duration_seconds':display_samples/RATE,
        'padding_seconds':(display_samples-samples)/RATE}


def installed_voices():
    result=[]
    for line in command(['/usr/bin/say','-v','?']).splitlines():
        match=re.match(r'^(.*?)\s+[a-z]{2}_[A-Z]{2}\s+#',line)
        if match:result.append(match[1].strip())
    return set(result)


def prepare(source,request,output,*,caption_records=None,voice='Samantha (English (US))',rate=180,dry_run=False):
    source,output=Path(source),Path(output);validate_media(source,request);guard(output)
    if output.exists():raise ValueError('preserve existing narration')
    if type(rate) is not int or not 80<=rate<=300:raise ValueError('explicit TTS rate between80and300 required')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise RuntimeError('FFmpeg required')
    native=request.media_kind in {'video','audio'}
    if native:
        info=probe(source);audio=[s for s in info['streams'] if s['codec_type']=='audio']
        if len(audio)>1:raise ValueError('ambiguous multiple audio tracks')
        duration=float(info['format']['duration'])
        if duration<=0 or duration>1800:raise ValueError('unsupported native duration')
        if audio:
            output.mkdir(parents=True)
            result={'schema':'numerical-narrated-presentation-v1','complete':True,'mode':'native_passthrough',
                'parent':request.model_dump(),'derivative':request.model_dump(),'media_path':str(source.resolve()),
                'duration_seconds':duration,'narration_generated':False,'original_clock_preserved':True,
                'both_branches_require_derivative_sha256':request.sha256,'schedule':None}
            (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
        if request.media_kind=='audio':raise ValueError('no decodable native audio track')
    if voice not in installed_voices():raise ValueError('voice must already be installed locally')
    output.mkdir(parents=True);pages=output/'pages';pages.mkdir()
    records=[]
    def description(index,images,sampling=None):
        if any(r.get('source_page_index')==index for r in (caption_records or [])):
            return caption(caption_records,index,request.sha256)
        if dry_run:raise ValueError('dry run needs cached caption; automatic visual description was not called')
        from .captioning import describe
        record=describe(source,request,images,index,output/f'caption-{index:04d}',sample_metadata=sampling)
        return record['text'],record
    if request.media_kind=='text':
        text=source.read_text(encoding='utf-8');records=[{'source_page_index':0,'text':text,'text_basis':'verbatim UTF-8 source','image':None}]
    elif request.media_kind=='pdf':
        for tool in ('pdfinfo','pdftoppm','pdftotext'):
            if not shutil.which(tool):raise RuntimeError('Poppler required for narratedPDF')
        found=re.search(r'^Pages:\s+(\d+)',command(['pdfinfo',str(source)]),re.M)
        if not found or not 1<=int(found[1])<=100:raise ValueError('PDF page limit or count invalid')
        for index in range(int(found[1])):
            guard(output);prefix=pages/f'page-{index:04d}'
            command(['pdftoppm','-f',str(index+1),'-l',str(index+1),'-singlefile','-scale-to','1920','-png',str(source),str(prefix)])
            text=command(['pdftotext','-f',str(index+1),'-l',str(index+1),'-enc','UTF-8',str(source),'-']).replace('\f','').strip()
            proof=None
            if not text:text,proof=description(index,[prefix.with_suffix('.png')])
            records.append({'source_page_index':index,'text':text,'text_basis':'generated descriptive caption' if proof else 'Poppler extraction; reading order unverified',
                'caption_provenance':proof,'image':str(prefix.with_suffix('.png').resolve())})
    elif request.media_kind=='image':
        from PIL import Image,ImageOps
        image=pages/'page-0000.png'
        with Image.open(source) as original:
            if getattr(original,'n_frames',1)!=1:raise ValueError('animated image requiresvideo')
            rgba=ImageOps.exif_transpose(original).convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba);bg.convert('RGB').save(image)
        text,proof=description(0,[image])
        records=[{'source_page_index':0,'text':text,'text_basis':'generated descriptive caption','caption_provenance':proof,'image':str(image.resolve())}]
    else:
        samples=[];sampling=[]
        if not caption_records:
            for index in range(6):
                second=duration*(index+.5)/6;image=pages/f'silent-sample-{index:04d}.png'
                command(['ffmpeg','-nostdin','-v','error','-n','-ss',str(second),'-i',str(source),'-frames:v','1',
                    '-vf','scale=1920:1080:force_original_aspect_ratio=decrease','-threads','1',str(image)])
                samples.append(image);sampling.append({'requested_seek_seconds':second,'native_frame_timestamp_established':False})
        text,proof=description(0,samples,sampling)
        records=[{'source_page_index':0,'text':text,'text_basis':'generated descriptive caption of silent video','caption_provenance':proof,'image':None}]
    if any(not r['text'].strip() or len(r['text'])>100000 for r in records):raise ValueError('empty or oversized narration text')
    plan={'schema':'numerical-narrated-presentation-v1','parent':request.model_dump(),'voice':voice,'rate_words_per_minute':rate,
        'narration_generated':True,'participant_data_used':False,'pages':records,'complete':False,'dry_run':dry_run}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if dry_run:return plan
    is_video=request.media_kind!='text';cursor=0
    for index,record in enumerate(records):
        guard(output);textpath=pages/f'page-{index:04d}.txt';textpath.write_text(record['text'])
        aiff=pages/f'page-{index:04d}.aiff';wav=pages/f'page-{index:04d}.wav'
        command(['/usr/bin/say','-v',voice,'-r',str(rate),'-f',str(textpath),'-o',str(aiff)],timeout=900)
        command(['ffmpeg','-nostdin','-v','error','-n','-i',str(aiff),'-ac','1','-ar',str(RATE),'-c:a','pcm_s16le',str(wav)])
        with wave.open(str(wav)) as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(RATE,1,2):raise ValueError('narration PCM mismatch')
            slot=clock(w.getnframes(),video=is_video)
        record.update(**slot,start=cursor/RATE,audio_file=str(wav.resolve()),audio_sha256=file_sha256(wav),text_sha256=file_sha256(textpath))
        cursor+=slot['display_samples']
        if cursor/RATE>1800:raise ValueError('narrated presentation exceeds30minutes; stop before remaining TTS')
    if cursor/RATE>1800:raise ValueError('narrated presentation exceeds30minutes')
    if native and cursor/RATE>duration+1e-6:raise ValueError('narration exceeds fixed silent-video slot; unsupported without altering original clock')
    combined=output/'narration.wav'
    with wave.open(str(combined),'wb') as writer:
        writer.setparams((1,2,RATE,0,'NONE','not compressed'))
        for r in records:
            guard(output)
            with wave.open(r['audio_file'],'rb') as reader:
                while data:=reader.readframes(RATE):writer.writeframes(data)
            writer.writeframes(bytes((r['display_samples']-r['spoken_samples'])*2))
    if not is_video:media=combined;media_kind='audio';actual=cursor/RATE
    else:
        media=output/'presentation.mp4';media_kind='video'
        if native:
            command(['ffmpeg','-nostdin','-v','error','-n','-i',str(source),'-i',str(combined),'-map','0:v:0','-map','1:a:0','-c:v','copy','-af','apad','-t',str(duration),'-c:a','aac','-movflags','+faststart',str(media)],timeout=1800)
            actual=duration
        else:
            concat=['ffconcat version 1.0']
            for r in records:concat.extend([f"file 'page-{r['source_page_index']:04d}.png'",f"duration {r['duration_seconds']:.8f}"])
            concat.append(f"file 'page-{records[-1]['source_page_index']:04d}.png'")
            listing=pages/'presentation.ffconcat';listing.write_text('\n'.join(concat)+'\n')
            command(['ffmpeg','-nostdin','-v','error','-n','-f','concat','-safe','1','-i',str(listing),'-i',str(combined),'-map','0:v:0','-map','1:a:0','-vf','scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1,fps=25','-t',str(cursor/RATE),'-c:v','libx264','-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-threads','1','-c:a','aac','-movflags','+faststart',str(media)],timeout=1800)
            actual=cursor/RATE
        decoded=probe(media)
        if len([s for s in decoded['streams'] if s['codec_type']=='audio'])!=1 or abs(float(decoded['format']['duration'])-actual)>1/FPS+1e-6:raise ValueError('narrated derivative clock mismatch')
    derivative=request.model_copy(update={'source_id':'narrated-'+request.source_id[:90],'sha256':file_sha256(media),'size_bytes':media.stat().st_size,'media_kind':media_kind})
    derivative=PublicMediaRequest.model_validate(derivative.model_dump());validate_media(source,request)
    result={**plan,'complete':True,'mode':'narrated_simulation','derivative':derivative.model_dump(),'media_path':str(media.resolve()),
        'duration_seconds':actual,'original_clock_preserved':native,'both_branches_require_derivative_sha256':derivative.sha256,
        'schedule':{'clock_basis':'actual local TTS audio duration; <=40ms explicit page-end padding for25fps rendering','pages':[{'source_page_index':r['source_page_index'],'start':r['start'],'duration':r['duration_seconds'],'spoken_seconds':r['spoken_seconds'],'padding_seconds':r['padding_seconds']} for r in records]},
        'narration_wav_sha256':file_sha256(combined),'native_branches_must_decode_final_presentation_audio':is_video,
        'neural_predictions_are_for_generated_presentation':True,'silent_reading_or_gaze_simulated':False}
    (output/'request.json').write_text(derivative.model_dump_json(indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
