from typing import DefaultDict, Union, Callable, Dict
from collections import Collection, Mapping
import numpy as np
import hashlib


def rec_map(data: Collection, func: Callable) -> Collection:
    result = func(data)
    if result is not None:
        return result

    _apply = lambda x: rec_map(x, func)
    if isinstance(data, Mapping):
        return type(data)({k: _apply(v) for k, v in data.items()})
    elif isinstance(data, Collection):
        return type(data)(_apply(v) for v in data)
    return data


try:
    import torch

    def md5(x: Union[torch.nn.Module, torch.Tensor, np.ndarray, bytes]) -> str:
        if isinstance(x, torch.nn.Module):
            x = torch.nn.utils.parameters_to_vector(x.parameters())
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        if isinstance(x, np.ndarray):
            x = x.tobytes()
        return hashlib.md5(x).digest().hex()
except ImportError:
    torch = None

    def md5(*args):
        raise NotImplementedError


class AverageMeter:
    count: int
    sum: float

    def __init__(self):
        self.count = 0
        self.sum = 0

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1e-3)

    def update(self, x, n=1):
        self.count += n
        self.sum += x * n

    def reset(self):
        self.count = 0
        self.sum = 0

    def __iadd__(self, x):
        self.update(x)
        return self


class MeterLib(DefaultDict[str, AverageMeter]):
    def __init__(self):
        super().__init__(AverageMeter)

    def purge(self, prefix='') -> Dict[str, float]:
        ret = {}
        for key, meter in self.items():
            if key.startswith(prefix):
                ret[key] = meter.mean
                meter.reset()
        return ret


def git_backup(zip_path):
    try:
        from git import Repo
        from git.exc import InvalidGitRepositoryError
    except ImportError as e:
        print(f"Can't import `git`: {e}")
        return

    try:
        repo = Repo('.')
        from zipfile import ZipFile
        pkg = ZipFile(zip_path, 'w')

        for file_name in repo.git.ls_files().split():
            pkg.write(file_name)

    except InvalidGitRepositoryError as e:
        print(f"Can't use git to backup files: {e}")
    except FileNotFoundError as e:
        print(f"Can't find file {e}. Did you delete a file and forget to `git add .`")


def set_random_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        if tf._major_api_version < 2:
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
    except ImportError:
        tf = None
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_breakpoint():
    import lunzi as lz
    import os

    try:
        # conflicts with `wandb` so we don't import it. Python will import it
        # import ipdb

        env_var = 'PYTHONBREAKPOINT'
        if env_var in os.environ:
            lz.log.critical(f'skip patching `ipdb`: environment variable `{env_var}` has been set.')
        else:
            os.environ[env_var] = 'ipdb.set_trace'
    except ImportError:
        lz.log.critical(f'skip patching `ipdb`: `ipdb` not found.')
