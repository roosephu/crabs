import torch
import wrapt
import numpy as np

import lunzi as lz


@wrapt.decorator
def _maybe_numpy(wrapped, instance, args, kwargs):
    device = next(instance.parameters()).device

    src = {'type': ''}

    def to_torch_type(x):
        if isinstance(x, np.ndarray):
            src['type'] = 'numpy'
            return torch.from_numpy(x).to(device)
        if isinstance(x, torch.Tensor):
            src['type'] = 'torch'
            src['device'] = x.device
            return x.to(device)
        return None

    def to_src_type(x):
        if isinstance(x, torch.Tensor):
            if src['type'] == 'numpy':
                return x.detach().cpu().numpy()
            if src['type'] == 'torch':
                return x.to(src['device'])
        return None

    with torch.no_grad():
        args = lz.utils.rec_map(args, to_torch_type)
        kwargs = lz.utils.rec_map(kwargs, to_torch_type)
        result = wrapped(*args, **kwargs)
        result = lz.utils.rec_map(result, to_src_type)
    return result


def maybe_numpy(fn):  # to remove PyCharm
    return _maybe_numpy(fn)


__all__ = ['maybe_numpy']
