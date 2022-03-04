from tempfile import NamedTemporaryFile
import datetime
import anymarkup
from itertools import product
from pathlib import Path
import asyncio


def grid(*items):
    items = list(items)
    for i in range(len(items)):
        if isinstance(items[i], dict):
            items[i] = grid(*[[{k: r} for r in v] if isinstance(v, (list, tuple)) else [{k: v}] for k, v in items[i].items()])
    assignments = list(product(*items))

    def combine(assignment):
        w = {}
        for p in assignment:
            w.update(p)
        return w

    assignments = [combine(assignment) for assignment in assignments]
    return assignments


class Task:
    def __init__(self, root_dir, template, mappings, *, shell='bash', pool_size=None, dry_run=False):
        self.shell = shell
        self.template = template
        if pool_size is not None:
            self.pool_size = min(len(mappings), pool_size)
        else:
            self.pool_size = len(mappings)

        uid = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        self.root_dir = Path(root_dir.format(uid=uid)).expanduser()
        self.root_dir.mkdir(exist_ok=False, parents=True)

        status = {'template': template, 'mappings': mappings}
        anymarkup.serialize_file(status, self.root_dir / '.status.json5')

        self.mappings = mappings
        self.scripts = [template.format(**mapping, root_dir=str(self.root_dir)) for mapping in mappings]

        self.tasks = []
        if not dry_run:
            self.run()

    def run(self):
        semaphore = asyncio.Semaphore(self.pool_size)

        async def execute(script):
            async with semaphore:
                with NamedTemporaryFile() as fp:
                    fp.write(script.encode())
                    fp.flush()
                    proc = await asyncio.create_subprocess_exec(
                        *self.shell.split(' '), fp.name,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    await proc.wait()
                    return proc

        self.tasks = [asyncio.create_task(execute(script)) for script in self.scripts]

    def cancel(self):
        for task in self.tasks:
            task.cancel()

    def __repr__(self):
        return f'Task(root = {str(self.root_dir)}, pool = {self.pool_size})'

    async def stdout(self, id=0):
        return await self.tasks[id].result().stdout.read()

    async def stderr(self, id=0):
        return await self.tasks[id].result().stderr.read()


__all__ = ['Task', 'grid']
