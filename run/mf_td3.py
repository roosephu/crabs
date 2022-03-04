import gym
import numpy as np
import torch
import torch.nn as nn

import lunzi as lz
from rlz import MLP
import rlz
import wandb
import pickle

from rlz.runner import SimpleRunner, EpisodeReturn, merge_episode_stats, ExtractLastInfo
import safe.envs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FLAGS(lz.BaseFLAGS):
    class env(lz.BaseFLAGS):
        id = 'HalfCheetah-v2'
        config = {}

    class ckpt(lz.BaseFLAGS):
        policy = ''

    TD3 = rlz.TD3Trainer.FLAGS
    n_steps = 1_000_000
    n_eval_freq = 10_000
    n_eval_samples = 12_000

    violation_penalty = 0

    min_pool = 10_000
    n_exploration_steps = 10_000


@FLAGS.set_defaults
def evaluate(step, runner, policy, tag, *, n_eval_samples):
    runner.reset()
    ep_infos = runner.run(policy, n_eval_samples)

    for key, value in merge_episode_stats(ep_infos).items():
        value = np.array(value)
        mean, std = np.mean(value), np.std(value)
        lz.log.info(f'# {step}, tag = {tag}, {key} = {mean:.6f} ± {std:.6f} over {len(value)} episodes.')
        wandb.log({
            'step': step,
            f'{tag}.{key}.mean': mean,
            f'{tag}.{key}.std': std,
            f'{tag}.{key}.n': len(value),
        })


@FLAGS.env.set_defaults
def make_env(*, id, config):
    import rlz.wrappers as w

    env = gym.make(id, **config)
    env = w.RescaleAction(env, -1, 1)
    env = w.ClipAction(env)
    env.seed(np.random.randint(0, 2**30))
    return env


class DetMLPPolicy(MLP, rlz.DetNetPolicy):
    pass


class TanhGaussianMLPPolicy(rlz.policy.TanhGaussianPolicy, MLP, rlz.NetPolicy):
    pass


class MLPQFn(MLP, rlz.NetQFn):
    pass


def main():
    lz.init(FLAGS)
    FLAGS.env.config['violation_penalty'] = FLAGS.violation_penalty
    env = make_env()

    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    make_stats = lambda: [EpisodeReturn(), ExtractLastInfo('episode.unsafe')]
    runners = {
        'explore': SimpleRunner(make_env, 1, make_stats, device=device),
        'evaluate': SimpleRunner(make_env, 1, make_stats, device=device),
    }
    buffer = rlz.ReplayBuffer(env, max_buf_size=FLAGS.n_steps)

    # hidden_sizes = [256, 256]
    hidden_sizes = [64, 64]
    qfn1 = MLPQFn([dim_state + dim_action, *hidden_sizes, 1]).to(device)
    qfn2 = MLPQFn([dim_state + dim_action, *hidden_sizes, 1]).to(device)
    # policy = DetMLPPolicy([dim_state, *hidden_sizes, dim_action], output_activation=nn.Tanh, auto_squeeze=False)\
    #     .to(device)
    # algo = rlz.TD3Trainer(policy, [qfn1, qfn2], sampler=buffer.sample, device=device)
    policy = TanhGaussianMLPPolicy([dim_state, *hidden_sizes, dim_action * 2]).to(device)
    mean_policy = rlz.policy.MeanPolicy(policy)
    algo = rlz.SACTrainer(policy, [qfn1, qfn2], sampler=buffer.sample, target_entropy=-dim_action, device=device)

    if FLAGS.ckpt.policy != '':  # must be done before define policy_optimizer (policy target init)
        policy.load_state_dict(torch.load(FLAGS.ckpt.policy, map_location=device)['policy'])
        lz.log.info(f"Load policy from {FLAGS.ckpt.policy}")

    expl_policy = rlz.policy.UniformPolicy(dim_action)
    for T in range(FLAGS.n_steps):
        if T == FLAGS.n_exploration_steps:
            # expl_policy = rlz.policy.AddGaussianNoise(policy, 0, 0.1)
            expl_policy = policy
            lz.log.warning('Switch to optimized policy')
        if T % FLAGS.n_eval_freq == 0:
            evaluate(T, runners['evaluate'], mean_policy, 'mean policy')
            # evaluate(T, runners['evaluate'], policy, 'policy')
            torch.save({
                'qfn1': qfn1.state_dict(),
                'qfn2': qfn2.state_dict(),
                'policy': policy.state_dict(),
            }, lz.log_dir / f'ckpt-{T}.pt')
            # with open(lz.log_dir / f"buf-{T}.pt", 'wb') as f:
            #     pickle.dump(buffer, f)
        ep_info = runners['explore'].run(expl_policy, 1, buffer=buffer)
        if len(ep_info) and 'episode.unsafe' in ep_info[0]:
            wandb.log({'step': T, 'expl_unasfe': ep_info[0]['episode.unsafe']})
            n_expl_unsafe_states = int(ep_info[0]['episode.unsafe'])
            returns = ep_info[0]['return']
            lz.log.info(f"[explore] # {T}: # expl unsafe states = {n_expl_unsafe_states}, return = {returns:.6f}")

        if T >= FLAGS.min_pool:
            algo.step()


if __name__ == '__main__':
    main()
