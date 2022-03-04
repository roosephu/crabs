from . import classic_pendulum
from . import inverted_pendulum


import lunzi as lz
import gym
import numpy as np


class FLAGS(lz.BaseFLAGS):
    id = 'MyPendulum-v0'
    config = {}


@FLAGS.set_defaults
def make_env(*, id, config):
    import rlz.wrappers as w

    env = gym.make(id, **config)
    env = w.RescaleAction(env, -1, 1)
    env = w.ClipAction(env)
    env = w.CastDtype(env)
    env.seed(np.random.randint(0, 2**30))
    return env
