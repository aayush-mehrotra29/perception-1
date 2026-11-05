"""Cache only content speech projections before native replay finishes."""
from pathlib import Path
import argparse,json,time
from .waveform import pcm44100,speech_projected,PCA_SHA,write
from brain.worker_contract import file_sha256
from brain.edugage_targets import guard

def run(audio,output):
    audio,output=map(Path,(audio,output));guard(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();digest=file_sha256(audio);pcm,derivative=pcm44100(audio,output/'canonical.wav')
    speech_projected(pcm,digest,output)
    m={'schema':'numerical-speech-cache-v1','complete':True,'audio_source_sha256':digest,'audio_path':str(audio.resolve()),
       'pcm_frames':len(pcm),'pca_sha256':PCA_SHA,'canonical_audio':derivative,'elapsed_seconds':time.monotonic()-start,
       'files':{p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}}
    write(output/'manifest.json',m);return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('output');a=p.parse_args();print(json.dumps(run(a.audio,a.output)))
