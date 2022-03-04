from typing import Union
from pathlib import Path
from .utils import git_backup

import fasteners
import anymarkup


root_dir = Path('./lunzi_logdir').expanduser()
log_dir = Path('./lunzi_logdir').expanduser()


def _read_exp_status(_exp_dir):
    status_path = _exp_dir / '.status.json5'
    if status_path.exists():
        return anymarkup.parse_file(status_path)
    else:
        run_id = 0
        while (root_dir / str(run_id)).exists():
            run_id += 1
        return {'id': run_id}


def init(_exp_dir: Union[str, Path]):
    global log_dir, root_dir

    root_dir = Path(_exp_dir)
    with fasteners.InterProcessLock(root_dir / '.lock'):
        _exp = _read_exp_status(root_dir)
        run_id = str(_exp.get('id', 0))
        _exp['id'] = _exp.get('id', 0) + 1

        log_dir = Path(root_dir).expanduser() / run_id
        log_dir.mkdir(parents=True, exist_ok=True)

        anymarkup.serialize_file(_exp, root_dir / '.status.json5')

    git_backup(log_dir)
    import lunzi as lz
    lz.log_dir = log_dir
