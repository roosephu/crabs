import os
import anymarkup
from pathlib import Path

import lunzi as lz
from .flags import MetaFLAGS, parse
from .utils import git_backup, set_random_seed, set_breakpoint
import wandb


@lz.timer
def init(root: MetaFLAGS):
    if 'seed' not in root:
        root.add('seed', int.from_bytes(os.urandom(3), 'little'))

    args = parse(root)
    root.freeze()

    root_dir = Path(args.root_dir).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(dir=str(root_dir), config=root.to_dict(), save_code=True)
    wandb.config.root_dir = root_dir

    lz.log_dir = Path(wandb.run.dir).parent
    wandb.config.log_dir = lz.log_dir

    seed = root.seed
    set_random_seed(seed)
    set_breakpoint()

    # don't use `wandb.run.log_code()` as it doesn't upload the the created artifact.
    git_backup(lz.log_dir / 'files' / 'source.zip')

    anymarkup.serialize_file(root.to_dict(), lz.log_dir / 'files' / 'config.json5')
    if args.print_config:
        serialized = anymarkup.serialize(root.to_dict(), 'json5').decode('utf-8')
        lz.log.info(f'FLAGS: {serialized}')

    lz.log.warning(f'log dir     = {str(lz.log_dir)}')
    lz.log.warning(f'output      = {str(lz.log_dir / "files" / "output.log")}')
    lz.log.warning(f'config      = {str(lz.log_dir / "files" / "config.json5")}')
    if not wandb.run.offline:
        lz.log.warning(f"wandb url   = {wandb.run.url}")

    # see https://hpcc.umd.edu/hpcc/help/slurmenv.html
    if 'SLURM_JOB_ID' in os.environ:
        lz.log.warning(f"Slurm JobID = {os.environ['SLURM_JOB_ID']}")

