import numpy as np
import torch
import torch.utils.data.dataset


class ReplayBuffer(torch.utils.data.dataset.Dataset):
    def __init__(self, env, max_buf_size: int, device='cpu'):
        self.env = env
        self.device = torch.device(device)
        self.max_buf_size = max_buf_size
        self.data = None
        self.length = 0

    def process_transition(self, transitions: dict, *, batch: bool): # -> dict[str, torch.Tensor]:
        ret = {}
        for key in transitions.keys():
            value = transitions[key]
            if key != 'info':
                ret[key] = torch.as_tensor(value)
        return ret

    def _init_like(self, transition, batch=True):
        self.data = {}
        for key in transition.keys():
            # TorchReplayBuffer doesn't support .items()
            value = transition[key]
            shape, dtype = value.shape, value.dtype
            if batch:
                shape = shape[1:]

            self.data[key] = torch.empty(self.max_buf_size, *shape, dtype=dtype, device=self.device)

    def add_transition(self, transition: dict):
        transition = self.process_transition(transition, batch=False)

        if self.data is None:
            self._init_like(transition, batch=False)

        for key in self.data.keys():
            self._buf_add(self.data[key], self.length % self.max_buf_size, transition[key], batch=False)
        self.length += 1

    def sample(self, n_samples: int = 1, *, indices=None):
        if indices is None:
            indices = np.random.randint(len(self), size=(n_samples,), dtype=np.int64)
        batch = {k: v[indices] for k, v in self.data.items()}
        return batch

    def __len__(self):
        return min(self.length, self.max_buf_size)

    @torch.no_grad()
    def add_transitions(self, transitions):
        if isinstance(transitions, (dict, ReplayBuffer)):
            if isinstance(transitions, dict):
                transitions = self.process_transition(transitions, batch=True)
                lengths = [len(v) for v in transitions.values()]
                n = lengths[0]
                assert all([length == n for length in lengths])
            else:
                n = len(transitions)

            if self.data is None:
                self._init_like(transitions, batch=True)

            # self.data might not be initialized yet.
            for key in transitions.keys():
                # for i in range(n):
                #     self._buf_add(self.data[key], (self.length + i) % self.max_buf_size, transitions[key][i])
                self._buf_add(self.data[key], self.length % self.max_buf_size, transitions[key], batch=True)
            self.length += n
        elif isinstance(transitions, (list, tuple, np.recarray)):
            for transition in transitions:
                self.add_transition(transition)
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item][:len(self)]

    def keys(self):
        return self.data.keys()

    def data_loader(self, batch_size, *, n_iters_per_epoch=None, replace=False):
        from torch.utils.data.dataloader import DataLoader
        buf = self

        class Loader:
            def __iter__(self):
                assert buf.data is not None
                if replace:
                    n_iters = n_iters_per_epoch if n_iters_per_epoch is not None else len(buf) // batch_size
                    for _ in range(n_iters):
                        yield buf.sample(batch_size)
                else:
                    assert n_iters_per_epoch is None
                    n = len(buf)
                    indices = np.random.permutation(len(buf))
                    for i in range(0, n, batch_size):
                        yield buf.sample(indices=indices[i:i + batch_size])

        return Loader()

    def __iadd__(self, other):
        self.add_transitions(other)
        return self

    def _buf_add(self, buf, idx, data, *, batch):
        if batch:
            n = len(data)
            if n + idx > self.max_buf_size:
                m = self.max_buf_size - idx
                buf[idx:] = data[:m]
                buf[:n - m] = data[m:]
            else:
                buf[idx:idx+n] = torch.as_tensor(data, device=self.device)
        else:
            buf[idx] = torch.as_tensor(data, device=self.device)
