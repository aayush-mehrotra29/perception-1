"""Phase-preserving readout experiment; the original model stays unchanged."""
import numpy as np
import torch
from .model import TemporalEEG


class EndpointEEG(TemporalEEG):
    def forward(self, x, progress):
        h = torch.nn.functional.gelu(self.project(x))
        for d, conv in zip(self.dilations, self.layers):
            h = h[..., (self.kernel - 1) * d:] + torch.nn.functional.gelu(conv(h))
        return self.head(torch.cat([h[..., -1], progress], dim=-1))


def load_endpoint(path, inputs, outputs, architecture):
    model = EndpointEEG(inputs, outputs, architecture)
    with np.load(path, allow_pickle=False) as z:
        model.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files}, strict=True)
    return model.eval()


def choose_blend(rows):
    """Development-only rule: MSE must not worsen; then maximize signed r.

    Zero weight remains a baseline fallback, never a multimodal success.
    Rows contain equal-clip metrics for each predefined weight.
    """
    baseline = next(r for r in rows if r['weight'] == 0)
    eligible = [r for r in rows if r['MSE'] <= baseline['MSE']
                and r['macro_r'] is not None]
    return max(eligible, key=lambda r: (r['macro_r'], -r['weight']))
