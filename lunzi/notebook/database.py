import warnings

import numpy as np
from typing import Union, Any
from pathlib import Path

import pandas as pd

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

import json


def flatten_dict(obj, sep='.'):
    out = {}

    def dfs(o, prefix):
        for key, value in o.items():
            if isinstance(value, dict):
                dfs(value, prefix + key + sep)
            else:
                out[prefix + key] = value

    dfs(obj, '')
    return out


def _flatten(proto):
    return flatten_dict({proto.key: json.loads(proto.value_json)})


class WandbRun:
    history: dict[str, pd.DataFrame]
    config: dict[str, Any]
    summary: dict[str, Any]

    def __init__(self, filename):
        self.filename = filename
        self.load()

    def load(self):
        store = datastore.DataStore()
        store.open_for_scan(self.filename)

        history = {}
        self.config = {}
        self.summary = {}

        records = {}
        while True:
            data = store.scan_data()
            if data is None:
                break
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)
            record_type = pb.WhichOneof("record_type")
            records[record_type] = records.get(record_type, 0) + 1
            if record_type == 'history':
                _runtime, _timestamp, _step, data = None, None, None, {}
                for item in pb.history.item:
                    p = pb
                    key, value = item.key, json.loads(item.value_json)
                    if item.key == '_runtime':
                        _runtime = value
                    elif item.key == '_timestamp':
                        _timestamp = value
                    elif item.key == '_step':
                        _step = value
                    else:
                        data.update(flatten_dict({key: value}))

                for key, value in data.items():
                    if key not in history:
                        history[key] = []
                    history[key].append({'_step': _step, key: value})

            elif record_type == 'telemetry':
                pass
            elif record_type == 'footer':
                pass
            elif record_type == 'header':
                pass
            elif record_type == 'final':
                pass
            elif record_type == 'output':
                pass
            elif record_type == 'run':
                run = pb.run
                self.id = run.run_id
                self.project = run.project
                if run.HasField('config'):
                    self._update_config(run.config)
            elif record_type == 'config':
                self._update_config(pb.config)
            elif record_type == 'files':
                pass
            elif record_type == 'stats':
                pass
            elif record_type == 'exit':
                pass
            elif record_type == 'summary':
                for update in pb.summary.update:
                    for key, value in _flatten(update).items():
                        self.summary[key] = value

        store.close()

        for key in history:
            history[key] = pd.DataFrame.from_records(history[key])

        self.history = history
        self.records = records

    def _update_config(self, pb):
        for update in pb.update:
            for key, value in _flatten(update).items():
                self.config[key] = value

    def metrics(self, keys: Union[str, list[str]]) -> pd.DataFrame:
        if isinstance(keys, str):
            return self.history[keys]
        output = self.history[keys[0]]
        for key in keys[1:]:
            output = output.merge(self.history[key], on='_step')
        return output

    def __getitem__(self, keys: Union[str, list[str]]) -> Union[np.ndarray, list[np.ndarray]]:
        if isinstance(keys, str):
            return self.history[keys][keys].to_numpy()
        metrics = self.metrics(keys)
        return [metrics[key].to_numpy() for key in keys]


class Store:
    def __init__(self):
        self._cache = {}
        self.verbose = False

    def clear(self):
        self._cache = {}

    def run(self, filename, reload=False):
        query = str(filename)
        if query not in self._cache or reload:
            if self.verbose:
                print(f'Loading {filename}')
            self._cache[query] = WandbRun(filename)
        return self._cache[query]

    def runs(self, *root_dirs, reload=False):
        import struct

        ret = []
        for root_dir in root_dirs:
            root_dir = Path(root_dir).expanduser()
            for filename in root_dir.glob('**/*.wandb'):
                try:
                    ret.append(self.run(filename, reload=reload))
                except struct.error as e:
                    warnings.warn(f"Can't parse {filename} {e}")
        return ret


db = Store()


__all__ = ['db', 'WandbRun']

