from typing import Union, Any, Dict, Callable
from pathlib import Path
from argparse import ArgumentParser
from loguru import logger
from inspect import signature, Parameter

import anymarkup
import wrapt


class MetaFLAGS(type):
    _frozen = False
    _strict = True
    seed: int

    def __setattr__(cls, key: str, value: Any):
        # assert not cls._frozen, 'Modifying frozen FLAGS.'
        super().__setattr__(key, value)

    def __getitem__(cls, item: str) -> Any:
        return cls.__dict__[item]

    def is_strict(cls) -> bool:
        return cls._strict

    def add(cls, key: str, value: Any, overwrite=False, overwrite_false=False):
        if key not in cls or overwrite or not getattr(cls, key) and overwrite_false:
            setattr(cls, key, value)

    def __iter__(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not isinstance(value, classmethod):
                if isinstance(value, MetaFLAGS):
                    value = value.to_dict()
                yield key, value

    def to_dict(cls) -> Dict[str, Any]:
        return dict(iter(cls))

    def freeze(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and isinstance(value, MetaFLAGS):
                value.freeze()
        cls.finalize()
        cls._frozen = True

    def finalize(cls):
        pass

    @property
    def set_defaults(cls):
        def decorate(fn: Callable):
            parameters = signature(fn).parameters
            common_keys = parameters.keys() & cls.keys()
            for key in common_keys:
                assert parameters[key].kind == Parameter.KEYWORD_ONLY

            @wrapt.decorator
            def _wrap(wrapped, instance, args, kwargs):
                # easier for pdb... only need to step over one line
                new_kwargs = {key: cls[key] for key in common_keys}
                new_kwargs.update(kwargs)
                return wrapped(*args, **new_kwargs)

            wrap = _wrap(fn)
            wrap.__unwrapped__ = fn
            wrap.config = cls
            return wrap
        return decorate

    def merge(cls, other: dict):
        return merge(cls, other)

    def copy(cls) -> 'MetaFLAGS':  # not deepcopy!
        class C(BaseFLAGS):
            pass
        for key, value in cls.__dict__.items():
            setattr(C, key, value)
        C.__name__ = cls.__name__
        return C

    def __copy__(self):
        return self.copy()

    def __repr__(cls):
        import json5
        return json5.dumps(cls.to_dict(), indent=2)

    def keys(cls):
        return [k for k, v in cls.__dict__.items() if not k.startswith('_') and not isinstance(v, classmethod)]

    def add_to_parser(cls, parser: ArgumentParser, prefix: str = '', rec: bool = True, keys: list = None) -> list:
        if keys is None:
            keys = []
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not isinstance(value, classmethod):
                full_key = f'{prefix}{key}'
                if isinstance(value, MetaFLAGS) and rec:
                    value.add_to_parser(parser, full_key + '.', True, keys)
                elif value is None:
                    logger.warning(f"{full_key} has default value None. Skipped.")
                else:
                    # we defaults it to None. As documented, argparser won't try to parse None.
                    if isinstance(value, bool):
                        logger.warning(f"{full_key} has type of bool, which is non-recommended, "
                                       f"as `--{full_key} False` also set it True.")
                    parser.add_argument('--' + full_key, default=None, type=type(value), dest=full_key,
                                        help=f"Defaults to {repr(value)}. Generated by Lunzi. ", metavar='')
                    keys.append(full_key)
        return keys


class BaseFLAGS(metaclass=MetaFLAGS):
    pass


def _set_attr(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _del_attr(obj, key):
    if isinstance(obj, dict):
        del obj[key]
    else:
        delattr(obj, key)


def resolve(config, path=Path.cwd()):  # not exposed to a top-level function
    for key, value in config.items():
        if isinstance(value, (MetaFLAGS, dict)):
            resolve(value)

    if '_merge' in config:
        for c in config['_merge']:
            merge(config, c)
        _del_attr(config, '_merge')
    if '_load' in config:
        load = config['_load']
        if isinstance(load, dict):
            for f in load.values():
                merge(config, load_config_file(path / f), update=False)
        else:
            merge(config, load_config_file(path / load))
        _del_attr(config, '_load')


def load_config_file(path):
    path = Path(path).resolve().expanduser()
    content = anymarkup.parse_file(path)
    resolve(content, path.parent)
    return content


def merge(cls: Union[MetaFLAGS, dict], rhs: dict, *, update=True):  # merging two non-magic flags
    for key, value in rhs.items():
        if key.startswith('_'):
            continue

        keys = cls if isinstance(cls, dict) else cls.__dict__
        if key not in keys:
            assert not isinstance(cls, MetaFLAGS) or not cls.is_strict(), \
                f"`{cls}` is strict and key {key} doesn't exist"
            _set_attr(cls, key, value)
        elif isinstance(cls[key], (MetaFLAGS, dict)) and isinstance(value, dict):
            merge(cls[key], value, update=update)
        elif update:
            _set_attr(cls, key, value)
    return cls


def set_value(cls: Union[MetaFLAGS, dict], path: str, value: Any):
    path = path.split('.')
    obj = {}
    current_node = obj
    for i, key in enumerate(path[:-1]):
        current_node[key] = {}
        current_node = current_node[key]
    current_node[path[-1]] = value

    resolve(obj)
    merge(cls, obj)


def parse_string(s: str):
    import ast
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def parse(root: MetaFLAGS):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configurations or config files', nargs='*', action='append')
    parser.add_argument('--root_dir', help='the root directory to logs', default='')
    root_keys = root.add_to_parser(parser)

    args = parser.parse_args()
    args.print_config = True
    if args.config:
        for cmd in sum(args.config, []):
            cmd: str
            if '=' in cmd:
                path, value = cmd.split('=', maxsplit=1)
                value = parse_string(value)
            else:
                path = '_load'
                if cmd.endswith('.json5'):
                    value = cmd
                else:
                    value = f'configs/{cmd}.json5'
            set_value(root, path, value)

    args_dict = vars(args)
    for key in root_keys:
        if args_dict[key] is not None:
            # don't need to parse, as argparse has done it.
            set_value(root, key, args_dict[key])

    return args
