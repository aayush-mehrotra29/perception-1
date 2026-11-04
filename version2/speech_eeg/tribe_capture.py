"""Read-only capture of native pre-subject TRIBE transformer representations.

Stage-B adapter only. Requires an authenticated live model and source timeline;
synthetic hook tests do not qualify a real cloud inference run.
"""
from pathlib import Path
import numpy as np
from .common import sha


class TribeHiddenCapture:
    def __init__(self,destination,frequency_hz,maximum_bytes=64*1024**2,existing_directory=False):
        self.destination=Path(destination);self.frequency=float(frequency_hz)
        if not np.isfinite(self.frequency) or self.frequency<=0:raise ValueError('Invalid native frequency')
        self.maximum_bytes=maximum_bytes;self.written=0;self.pending=None;self.records=[]
        self.destination.mkdir(parents=True,exist_ok=existing_directory)

    def batch_hook(self,model,args):
        if self.pending is not None:raise ValueError('Previous batch has no hidden output')
        if len(args)!=1:raise ValueError('Unexpected native model input')
        self.pending=[{'start':float(s.start),'duration':float(s.duration)} for s in args[0].segments]
        return None

    def hidden_hook(self,module,args,output):
        if self.pending is None:raise ValueError('Missing native batch clock')
        a=output.detach().cpu().numpy()
        if a.ndim!=3 or a.shape[0]!=len(self.pending) or not np.isfinite(a).all():
            raise ValueError('Expected finite batch,time,hidden native transformer output')
        for s in self.pending:
            if abs(s['duration']*self.frequency-a.shape[1])>1e-6:
                raise ValueError('Native hidden timeline does not match segment')
        a=np.array(a,dtype=np.float32,copy=True)
        if self.written+a.nbytes>self.maximum_bytes:raise ValueError('Capture byte limit exceeded')
        p=self.destination/f'hidden-{len(self.records):04d}.npz'
        with p.open('xb') as f:np.savez(f,hidden=a)
        self.records.append({'file':p.name,'sha256':sha(p),'shape':list(a.shape),
            'segments':self.pending,'frequency_hz':self.frequency,'axes':'batch,time,hidden',
            'location':'native encoder output before low-rank/subject projection and BOLD pooling',
            'context':'entire native segment; nominal bins do not imply causal physiological timing',
            'requires_live_clock_qualification':True})
        self.written+=a.nbytes;self.pending=None
        return None  # Never substitute tensors or modify the native forward pass.

    def attach(self,native_model):
        if not hasattr(native_model,'encoder'):raise ValueError('Native transformer encoder missing')
        return [native_model.register_forward_pre_hook(self.batch_hook),
                native_model.encoder.register_forward_hook(self.hidden_hook)]
