from time import perf_counter
from collections import defaultdict
from dataclasses import dataclass
import wrapt


@dataclass
class _Counter:
    time_elapsed = 0.
    count = 0

    @property
    def average(self):
        return self.time_elapsed / max(self.count, 1)

    def __repr__(self):
        return f'{{time: {self.time_elapsed:.3e}, count: {self.count}, avg = {self.average:.3e}}}'


@wrapt.decorator
def decorator(wrapped, instance, args, kwargs):
    with TimerContext([wrapped.__module__ + ':' + wrapped.__qualname__]):
        return wrapped(*args, **kwargs)


class TimerContext:
    def __init__(self, keys):
        self.keys = keys

    def __enter__(self):
        self.t0 = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = perf_counter() - self.t0
        for key in self.keys:
            timer.pool[key].time_elapsed += dt
            timer.pool[key].count += 1


class Timer:
    """Usage:
        > @timer
        > def f(): pass

        > with timer('xxx'):
        >    pass

        > print(timer['xxx'], timer['f'])
    """

    def __init__(self):
        self.pool: dict[str, _Counter] = defaultdict(_Counter)

    def __call__(self, *args, **kwargs):
        if len(args) == 0 or isinstance(args[0], str):
            return TimerContext(args)
        return decorator(args[0])

    def __getitem__(self, item):
        return self.pool[item]

    def items(self):
        return self.pool.items()

    def keys(self):
        return self.pool.keys()

    def values(self):
        return self.pool.values()


timer = Timer()


__all__ = ['timer']
