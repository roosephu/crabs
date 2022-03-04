import lunzi as lz
import torch
import torch.nn as nn
from torch.nn.functional import relu, softplus
import numpy as np
import pickle
import pytorch_lightning as pl
import wandb
from typing import Callable

from rlz.runner import merge_episode_stats, SimpleRunner, EpisodeReturn, EpisodeLength, ExtractLastInfo
from copy import deepcopy
from safe import TransitionModel, SafeSACTrainer2, Normalizer, EnsembleModel, GatedTransitionModel

import safe.envs
from safe.envs import make_env
from rlz import MLP
import rlz


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ObjEval = Callable[[torch.Tensor], dict]


class CrabsCore(torch.nn.Module):
    class FLAGS(lz.BaseFLAGS):
        class obj(lz.BaseFLAGS):
            eps = 0.0
            neg_coef = 1.0

    def __init__(self, h, model, policy):
        super().__init__()
        self.h = h
        self.policy = policy
        self.model = model

    def u(self, states, actions=None):
        if actions is None:
            actions = self.policy(states)
        next_states = [self.model.models[idx](states, actions) for idx in self.model.elites]
        all_next_states = torch.stack(next_states)
        all_nh = self.h(all_next_states)
        nh = all_nh.max(dim=0).values
        return nh

    @lz.timer
    def obj_eval(self, s):
        h = self.h(s)
        u = self.u(s)

        # can't be 1e30: otherwise 100 + 1e30 = 1e30
        eps = self.FLAGS.obj.eps
        obj = u + eps
        mask = (h < 0) & (u + eps > 0)
        return {
            'h': h,
            'u': u,
            's': s,
            'obj': obj,
            'constraint': h,
            'mask': mask,
            'max_obj': (obj * mask).max(),
            'hard_obj': torch.where(h < 0, u + eps, -h - 1000)
        }


class StateBox:
    INF = 1e10

    def __init__(self, shape, s0, device, expansion=1.5):
        self._max = torch.full(shape, -self.INF, device=device)
        self._min = torch.full(shape, +self.INF, device=device)
        self.center = None
        self.length = None
        self.expansion = expansion
        self.device = device
        self.s0 = s0
        self.shape = shape

    @torch.no_grad()
    def find_box(self, h):
        s = torch.empty(10_000, *self.shape, device=self.device)
        count = 0
        for i in range(1000):
            self.fill_(s)
            inside = torch.where(h(s) < 0.0)[0]
            if len(inside) and (torch.any(s[inside] < self._min) or torch.any(s[inside] > self._max)):
                self.update(s[inside])
                count += 1
            else:
                break

    def update(self, data, logging=True):
        self._max = self._max.maximum(data.max(dim=0).values)
        self._min = self._min.minimum(data.min(dim=0).values)
        self.center = (self._max + self._min) / 2
        self.length = (self._max - self._min) / 2 * self.expansion  # expand the box
        if logging:
            lz.log.info(f"[StateBox] updated: max = {self._max.cpu()}, min = {self._min.cpu()}")

    @torch.no_grad()
    def reset(self):
        nn.init.constant_(self._max, -self.INF)
        nn.init.constant_(self._min, +self.INF)
        self.update(self.s0 + 1e-3, logging=False)
        self.update(self.s0 - 1e-3, logging=False)

    @torch.no_grad()
    def fill_(self, s):
        s.data.copy_((torch.rand_like(s) * 2 - 1) * self.length + self.center)

    def decode(self, s):
        return s * self.length + self.center


class Barrier(nn.Module):
    class FLAGS(lz.BaseFLAGS):
        ell_coef = 1.
        barrier_coef = 1

    def __init__(self, net, env_barrier_fn, s0):
        super().__init__()
        self.net = net
        self.env_barrier_fn = env_barrier_fn
        self.s0 = s0
        self.ell = softplus

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # return self.net(state) + self.barrier_fn(state) - self.net(self.s0) + self.barrier_fn(self.s0) - self.offset
        return self.ell(self.net(states) - self.net(self.s0[None])) * self.FLAGS.ell_coef \
               + self.env_barrier_fn(states) * self.FLAGS.barrier_coef - 1


class SSampleOptimizer(nn.Module):
    def __init__(self, obj_eval: ObjEval, state_box: StateBox):
        super().__init__()
        self.obj_eval = obj_eval
        self.s = nn.Parameter(torch.randn(100_000, *state_box.shape), requires_grad=False)
        self.state_box = state_box

    @torch.no_grad()
    def debug(self, *, step):
        self.state_box.fill_(self.s)
        s = self.s
        result = self.obj_eval(s)

        hardD_sample_s = result['hard_obj'].max().item()
        inside = (result['h'] <= 0).sum().item()
        lz.log.debug(f"[S sampler]: hardD = {hardD_sample_s:.6f}, inside = {inside}")


class SGradOptimizer(nn.Module):
    def __init__(self, obj_eval: ObjEval, state_box: StateBox):
        super().__init__()
        self.obj_eval = obj_eval
        self.z = nn.Parameter(torch.randn(10000, *state_box.shape), requires_grad=True)
        self.opt = torch.optim.Adam([self.z], lr=1e-3)
        self.state_box = state_box

    @property
    def s(self):
        return self.state_box.decode(self.z)

    def step(self):
        result = self.obj_eval(self.s)
        obj = result['hard_obj']
        loss = (-obj).mean()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss

    @torch.no_grad()
    def reinit(self):
        lz.log.debug("[SGradOpt] reinit")
        nn.init.uniform_(self.z, -1., 1.)

    def debug(self, *, step):
        result = self.obj_eval(self.s)
        hardD = result['hard_obj']
        h = result['constraint']
        u = result['obj']
        idx = hardD.argmax()
        max_obj = hardD.max().item()
        if max_obj > 0:
            method = lz.log.warning
        else:
            method = lz.log.debug
        method(f"[S grad opt] hardD = {max_obj:.6f}, h = {h[idx].item():.6f}, u = {u[idx].item():.6f}, "
               f"inside = {(h <= 0).sum().item()}, s = {self.s[idx].cpu().detach()}")

        return {
            'optimal': hardD.max().item(),
        }


@torch.enable_grad()
def constrained_optimize(fx, gx, x, opt, reg=0.0):  # \grad_y [max_{x: g_y(x) <= 0} f(x)]
    sum_fx = fx.sum()
    sum_gx = gx.sum()
    with torch.no_grad():
        df_x = torch.autograd.grad(sum_fx, x, retain_graph=True)[0]
        dg_x = torch.autograd.grad(sum_gx, x, retain_graph=True)[0]
        lambda_ = df_x.norm(dim=-1) / dg_x.norm(dim=-1).clamp(min=1e-6)

    opt.zero_grad()
    (fx - gx * lambda_ + reg).sum().backward()
    opt.step()

    return {'df': df_x, 'dg': dg_x}


class SLangevinOptimizer(nn.Module):
    class FLAGS(lz.BaseFLAGS):
        class temperature(lz.BaseFLAGS):
            max = 0.1
            min = 0.001

        class filter(lz.BaseFLAGS):
            top_k = 10000
            pool = False

        n_steps = 1
        method = 'grad'
        lr = 0.01
        batch_size = 1000
        extend_region = 0.0
        barrier_coef = 0.
        L_neg_coef = 1
        resample = False

        n_proj_iters = 10
        precond = False

    def __init__(self, core: CrabsCore, state_box: StateBox):
        super().__init__()
        self.core = core
        self.temperature = self.FLAGS.temperature.max
        self.state_box = state_box

        self.z = nn.Parameter(torch.zeros(self.FLAGS.batch_size, *state_box.shape, device=device), requires_grad=True)
        self.tau = nn.Parameter(torch.full([self.FLAGS.batch_size, 1], 1e-2), requires_grad=False)
        self.alpha = nn.Parameter(torch.full([self.FLAGS.batch_size], 3.0), requires_grad=False)
        self.opt = torch.optim.Adam([self.z])
        self.max_s = torch.zeros(state_box.shape, device=device)
        self.min_s = torch.zeros(state_box.shape, device=device)

        self.mask = torch.tensor([0], dtype=torch.int64)
        self.n_failure = torch.zeros(self.FLAGS.batch_size, dtype=torch.int64, device=device)
        self.n_resampled = 0

        self.adam = torch.optim.Adam([self.z], betas=(0, 0.999), lr=0.001)
        self.since_last_reset = 0
        self.reinit()

    @property
    def s(self):
        return self.state_box.decode(self.z)

    def reinit(self):
        # self.state_box.fill_(self.s)
        lz.log.debug("[SGradOpt] reinit")
        nn.init.uniform_(self.z, -1., 1.)
        nn.init.constant_(self.tau, 0.01)
        nn.init.constant_(self.alpha, 3.0)
        self.since_last_reset = 0

    def set_temperature(self, p):
        max = self.FLAGS.temperature.max
        min = self.FLAGS.temperature.min
        self.temperature = np.exp(np.log(max) * (1 - p) + np.log(min) * p)

    def pdf(self, z):
        s = self.state_box.decode(z)
        result = self.core.obj_eval(s)
        return result['hard_obj'] / self.temperature, result

    def project_back(self, should_print=False):
        for _ in range(self.FLAGS.n_proj_iters):
            with torch.enable_grad():
                h = self.core.h(self.s)
                loss = relu(h - 0.03)
                if (h > 0.03).sum() < 1000:
                    break
                self.adam.zero_grad()
                loss.sum().backward()
                self.adam.step()

    @torch.no_grad()
    def resample(self, f: torch.Tensor, idx):
        if len(idx) == 0:
            return
        new_idx = f.softmax(0).multinomial(len(idx), replacement=True)
        self.z[idx] = self.z[new_idx]
        self.tau[idx] = self.tau[new_idx]
        self.n_failure[idx] = 0
        self.n_resampled += len(idx)

    def step(self):
        self.since_last_reset += 1
        self.project_back()
        tau = self.tau
        a = self.z

        f_a, a_info = self.pdf(a)
        grad_a = torch.autograd.grad(f_a.sum(), a)[0]

        w = torch.randn_like(a)
        b = a + tau * grad_a + (tau * 2).sqrt() * w
        b = b.detach().requires_grad_()
        f_b, b_info = self.pdf(b)
        grad_b = torch.autograd.grad(f_b.sum(), b)[0]
        going_out = (a_info['h'] < 0) & (b_info['h'] > 0)
        lz.meters['opt_s/going_out'] += going_out.to(torch.float32).mean()

        lz.meters['opt_s/out_to_in'] += ((a_info['h'] > 0) & (b_info['h'] < 0)).sum().item() / self.FLAGS.batch_size

        with torch.no_grad():
            log_p_a_to_b = -w.norm(dim=-1)**2
            log_p_b_to_a = -((a - b - tau * grad_b)**2).sum(dim=-1) / tau[:, 0] / 4
            log_ratio = (f_b + log_p_b_to_a) - (f_a + log_p_a_to_b)
            ratio = log_ratio.clamp(max=0).exp()[:, None]
            sampling = torch.rand_like(ratio) < ratio
            b = torch.where(sampling & (b_info['h'][:, None] < 0), b, a)
            new_f_b = torch.where(sampling[:, 0], f_b, f_a)
            lz.meters['opt_s/accept'] += sampling.sum().item() / self.FLAGS.batch_size

            self.mask = torch.nonzero(new_f_b >= 0)[:, 0]
            if len(self.mask) == 0:
                self.mask = torch.tensor([0], dtype=torch.int64)

            self.z.set_(b)
            # alpha should be moved slower than tau, as tau * grad will be smaller after one step.
            # self.alpha.mul_(FLAGS.lr * (going_out.to(torch.float32) - 0.5) + 1).clamp_(1e-4, 1e4)
            self.tau.mul_(self.FLAGS.lr * (ratio - 0.574) + 1)  # .clamp_(max=1.0)
            if self.FLAGS.resample:
                self.n_failure[new_f_b >= -100] = 0
                self.n_failure += 1
                self.resample(new_f_b, torch.nonzero(self.n_failure > 1000)[:, 0])
        return {
            'optimal': a_info['hard_obj'].max().item(),
        }

    @torch.no_grad()
    def debug(self, *, step=0):
        result = self.core.obj_eval(self.s)
        h = result['h']
        hardD_s = result['hard_obj'].max().item()
        inside = (result['constraint'] <= 0).sum().item()
        cut_size = result['mask'].sum().item()

        geo_mean_tau = self.tau.log().mean().exp().item()
        max_tau = self.tau.max().item()
        wandb.log({
            'step': step,
            'opt_s.hardD': hardD_s,
            'opt_s.inside': inside / self.FLAGS.batch_size,
            'opt_s.P_accept': lz.meters['opt_s/accept'].mean,
        })

        h_inside = h.cpu().numpy()
        h_inside = h_inside[np.where(result['constraint'].cpu() <= 0)]
        h_dist = np.percentile(h_inside, [25, 50, 75]) if len(h_inside) else []
        lz.log.debug(f"[S Langevin]: temperature = {self.temperature:.3f}, hardD = {hardD_s:.6f}, "
                     f"inside/cut = {inside}/{cut_size}, "
                     f"tau = [geo mean {geo_mean_tau:.3e}, max {max_tau:.3e}], "
                     f"Pr[accept] = {lz.meters['opt_s/accept'].mean:.3f}, "
                     f"Pr[going out] = {lz.meters['opt_s/going_out'].mean:.3f}, "
                     f"h 25/50/75% = {h_dist}"
                     )
        lz.meters.purge('opt_s/')
        self.n_resampled = 0

        return {
            'inside': inside
        }


class ExplorationPolicy(nn.Module, rlz.BasePolicy):
    def __init__(self, policy, core: CrabsCore):
        super().__init__()
        self.policy = policy
        self.crabs = core
        self.last_h = 0
        self.last_u = 0

    @torch.no_grad()
    def forward(self, states: torch.Tensor):
        device = states.device
        assert len(states) == 1
        dist = self.policy(states)

        if isinstance(dist, rlz.distributions.TanhGaussian):
            mean, std = dist.mean, dist.stddev

            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10., device=device)
            actions = (mean + torch.randn([n, *mean.shape[1:]], device=device) * std * decay[:, None]).tanh()
        else:
            mean = dist
            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10., device=device)
            actions = mean + torch.randn([n, *mean.shape[1:]], device=device) * decay[:, None]

        all_u = self.crabs.u(states, actions).detach().cpu().numpy()
        if np.min(all_u) <= 0:
            index = np.min(np.where(all_u <= 0)[0])
            action = actions[index]
            lz.meters['expl/backup'] += index
        else:
            action = self.crabs.policy(states[0])
            lz.meters['expl/backup'] += n

        return action[None]

    @rlz.torch_utils.maybe_numpy
    def get_actions(self, states):
        return self(states)


def evaluate(step, runner, policy, tag, *, n_eval_samples):
    runner.reset()
    ep_infos = runner.run(policy, n_eval_samples)

    for key, value in merge_episode_stats(ep_infos).items():
        value = np.array(value)
        mean, std = np.mean(value), np.std(value)
        if key == 'episode.unsafe':
            if value.sum() > 0:
                lz.log.warning(f'# {step}, tag = {tag}, {key} = {mean:.6f} ± {std:.6f} over {len(value)} episodes.')
        else:
            lz.log.info(f'# {step}, tag = {tag}, {key} = {mean:.6f} ± {std:.6f} over {len(value)} episodes.')
        wandb.log({
            'step': step,
            'n_episodes': self.,
            f'{tag}.{key}.mean': mean,
            f'{tag}.{key}.std': std,
            f'{tag}.{key}.n': len(value),
        })


class BarrierCertOptimizer:
    class FLAGS(lz.BaseFLAGS):
        weight_decay = 1e-4
        lr = 0.0003
        lambda_2 = 'norm'
        locals = {}

    def __init__(self, h: Barrier, obj_eval: ObjEval, core_ref: CrabsCore, s_opt: SLangevinOptimizer,
                 state_box: StateBox, h_ref: Barrier = None):
        super().__init__()
        self.h = h
        self.obj_eval = obj_eval
        self.core_ref = core_ref
        self.s_opt = s_opt
        self.state_box = state_box
        self.h_ref = h_ref
        self.s_opt_sample = SSampleOptimizer(self.obj_eval, self.state_box).to(device)
        self.s_opt_grad = SGradOptimizer(self.obj_eval, self.state_box).to(device)

        self.since_last_update = 0
        self.opt = torch.optim.Adam(self.h.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)

    def step(self):
        for i in range(FLAGS.opt_s.n_steps):
            self.s_opt.step()
        s = self.s_opt.s.detach().clone().requires_grad_()
        result = self.obj_eval(s)
        mask, obj = result['mask'], result['obj']
        regularization = 0
        if self.h_ref is not None:
            regularization = regularization + (result['h'] - self.h_ref(s)).clamp(min=0.).mean() * 0.001

        if mask.sum() > 0:
            constrained_optimize(obj * mask / mask.sum(), result['constraint'], s, self.opt, reg=regularization)
            lz.meters['opt_h/update_prob'] += 1
            self.since_last_update = 0
        else:
            lz.meters['opt_h/update_prob'] += 0
            self.since_last_update += 1
        return result

    def debug(self, *, step=0):
        self.s_opt.debug(step=step)
        lz.log.debug(f"[h opt]: Pr[update] = {lz.meters['opt_h/update_prob'].mean:0f}")
        lz.meters.purge('opt_h/')

    def train(self) -> (bool, float):
        for _ in range(2000):
            self.s_opt.step()

        h_status = 'training'
        self.since_last_update = 0
        for t in range(20_000):
            if t % 1_000 == 0:
                lz.log.info(f"# iter {t}")

            result = self.step()
            if result['mask'].sum() > 0.0:
                h_status = 'training'

            if h_status == 'training' and self.since_last_update >= 1000:
                lz.log.info("resetting SGLD, entering observation period")
                self.state_box.find_box(self.core_ref.h)
                self.s_opt.reinit()
                h_status = 'observation-period'

            if h_status == 'observation-period' and self.since_last_update == 5_000:
                lz.log.info(f"win streak at {t} => find a new invariant => update ref. ")
                if t == 4_999:  # policy is too conservative, reduce safe constraint
                    return True, 2.0
                else:
                    return True, 1.2
        return False, 0.5

    def pretrain(self):
        # don't tune state box
        self.state_box.reset()
        lz.log.info("pretrain s...")
        for i in range(FLAGS.n_pretrain_s_iters):
            if i % 1_000 == 0:
                self.s_opt.debug(step=i)
            self.s_opt.step()

        self.h_ref = None
        for t in range(FLAGS.n_iters):
            if t % 1_000 == 0:
                lz.log.info(f"# iter {t}")
                self.check_by_sample(step=t)
                self.s_opt.debug(step=t)

                result = self.obj_eval(self.s_opt.s)
                hardD = result['hard_obj']

                lz.log.debug(f"[h opt]: optimal = {hardD.max().item():.6f}, "
                             f"Pr[update] = {lz.meters['opt_h/update_prob'].mean:0f}")
                lz.meters.purge('opt_h/')
            if t % 50_000 == 0 and t > 0:
                self.check_by_grad()

            self.step()

            if self.since_last_update > 2000 and self.s_opt.since_last_reset > 5000:
                self.state_box.reset()
                self.state_box.find_box(self.h)
                self.s_opt.reinit()

    def check_by_sample(self, *, step=0):
        self.s_opt_sample.debug(step=step)

    def check_by_grad(self):
        self.s_opt_grad.reinit()
        for i in range(10001):
            if i % 1000 == 0:
                self.s_opt_grad.debug(step=0)
            self.s_opt_grad.step()


class FLAGS(lz.BaseFLAGS):
    _strict = False

    class model(lz.BaseFLAGS):
        type = 'learned'
        n_ensemble = 5
        n_elites = 0
        frozen = False
        train = TransitionModel.FLAGS

    class ckpt(lz.BaseFLAGS):
        h = ''
        policy = ''
        models = ''
        buf = ''

    class h(lz.BaseFLAGS):
        type = 'learned'

    env = safe.envs.FLAGS
    SAC = SafeSACTrainer2.FLAGS
    lyapunov = Barrier.FLAGS
    opt_s = SLangevinOptimizer.FLAGS
    opt_h = BarrierCertOptimizer.FLAGS
    crabs = CrabsCore.FLAGS

    n_iters = 500000
    n_plot_iters = 10000
    n_eval_iters = 1000
    n_save_iters = 10000
    n_pretrain_s_iters = 10000
    task = 'train'


class DetMLPPolicy(MLP, rlz.DetNetPolicy):
    pass


class MLPQFn(MLP, rlz.NetQFn):
    pass


class TanhGaussianMLPPolicy(rlz.policy.TanhGaussianPolicy, MLP, rlz.NetPolicy):
    pass


class PolicyAdvTraining:
    class FLAGS(lz.BaseFLAGS):
        weight_decay = 1e-4
        lr = 0.0003

    def __init__(self, policy, s_opt, obj_eval):
        self.policy = policy
        self.s_opt = s_opt
        self.obj_eval = obj_eval
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)

        self.count = 0.0

    def step(self, freq):
        for i in range(FLAGS.opt_s.n_steps):  # opt s
            self.s_opt.step()

        self.count += 1
        while self.count >= freq:
            self.count -= freq
            result = self.obj_eval(self.s_opt.s)
            mask = result['mask']

            if mask.any():
                self.opt.zero_grad()
                loss = (result['obj'] * mask).sum() / mask.sum()
                loss.backward()
                self.opt.step()


class Crabs:
    FLAGS = FLAGS

    def __init__(self, make_env: Callable):
        # environment related stuff
        env = make_env()
        self.make_env = make_env
        self.env = env
        self.s0 = torch.tensor(env.reset(), device=device, dtype=torch.float32)
        self.dim_state = env.observation_space.shape[0]
        self.dim_action = env.action_space.shape[0]
        self.horizon = env.spec.max_episode_steps
        self.n_expl_episodes = 0
        lz.log.info(f"env: dim state = {self.dim_state}, dim action = {self.dim_action}")

        self.normalizer = Normalizer(self.dim_state, clip=1000).to(device)
        self.state_box = StateBox([self.dim_state], self.s0, device)
        self.state_box.reset()

        self.buf_real = rlz.ReplayBuffer(env, max_buf_size=1000_000)
        self.buf_dev = rlz.ReplayBuffer(env, max_buf_size=10_000)
        self.policy = TanhGaussianMLPPolicy([self.dim_state, 64, 64, self.dim_action * 2]).to(device)
        self.mean_policy = rlz.policy.MeanPolicy(self.policy)

        if FLAGS.model.type == 'GatedTransitionModel':
            make_model = lambda i: \
                GatedTransitionModel(self.dim_state, self.normalizer,
                                     [self.dim_state + self.dim_action, 256, 256, 256, 256, self.dim_state * 2],
                                     name=f'model-{i}')
            self.model = EnsembleModel([make_model(i) for i in range(FLAGS.model.n_ensemble)]).to(device)
            self.model_trainer = pl.Trainer(
                max_epochs=0, gpus=1, auto_select_gpus=True, default_root_dir=lz.log_dir,
                progress_bar_refresh_rate=0, checkpoint_callback=False)
        elif FLAGS.model.type == 'TransitionModel':
            make_model = lambda i: \
                TransitionModel(self.dim_state, self.normalizer,
                                [self.dim_state + self.dim_action, 256, 256, 256, 256, self.dim_state * 2],
                                name=f'model-{i}')
            self.model = EnsembleModel([make_model(i) for i in range(FLAGS.model.n_ensemble)]).to(device)
            self.model_trainer = pl.Trainer(
                max_epochs=0, gpus=1, auto_select_gpus=True, default_root_dir=lz.log_dir,
                progress_bar_refresh_rate=0, checkpoint_callback=False)
        else:
            assert False, f"unknown model type {FLAGS.model.type}"

        self.horizon = env.spec.max_episode_steps
        make_stats = lambda: [ExtractLastInfo('episode.unsafe'), EpisodeReturn(), EpisodeLength()]
        self.runners = {
            'explore': SimpleRunner(make_env, 1, make_stats, device=device),
            'evaluate': SimpleRunner(make_env, 1, make_stats, device=device),
            'test': SimpleRunner(make_env, 1, make_stats, device=device),
        }

        self.h = Barrier(nn.Sequential(self.normalizer, MLP([self.dim_state, 256, 256, 1])), env.barrier_fn, self.s0)\
            .to(device)

        self.core = CrabsCore(self.h, self.model, self.mean_policy)
        if FLAGS.model.frozen:
            self.model.requires_grad_(False)
            lz.log.warning(f"models are frozen!")

        # policy optimization
        self.load_from_ckpt()

        self.core_ref = deepcopy(self.core)

        self.qf1 = MLPQFn([self.dim_state + self.dim_action, 256, 256, 1])
        self.qf2 = MLPQFn([self.dim_state + self.dim_action, 256, 256, 1])

        self.policy_optimizer = SafeSACTrainer2(
            self.policy,
            [self.qf1, self.qf2],
            self.core.u,
            sampler=self.buf_real.sample,
            device=device,
            target_entropy=-self.dim_action,
        )

        self.state_box.find_box(self.core_ref.h)
        self.s_opt = SLangevinOptimizer(self.core, self.state_box).to(device)

        lz.log.debug(f"[normalizer]: mean = {self.normalizer.mean.cpu().numpy()}, "
                     f"std = {self.normalizer.std.cpu().numpy()}")

        self.n_samples_so_far = 0

        self.h_opt = BarrierCertOptimizer(self.h, self.core.obj_eval, self.core_ref, self.s_opt, self.state_box)
        self.policy_adv_opt = PolicyAdvTraining(self.policy, self.s_opt, self.core.obj_eval)

    def run(self):
        getattr(self, f'_task_{FLAGS.task}')()

    def load_from_ckpt(self):
        if FLAGS.ckpt.policy != '':  # must be done before define policy_optimizer (policy target init)
            self.policy.load_state_dict(torch.load(FLAGS.ckpt.policy, map_location=device)['policy'])
            lz.log.info(f"Load policy from {FLAGS.ckpt.policy}")
        if FLAGS.ckpt.h != '' and FLAGS.h.type == 'learned':  # must be done before L_target is init
            self.h.load_state_dict(torch.load(FLAGS.ckpt.h, map_location=device)['h'])
            lz.log.info(f"Load h from {FLAGS.ckpt.h}")
        if FLAGS.ckpt.models != '':
            self.model.load_state_dict(torch.load(FLAGS.ckpt.models, map_location=device)['models'])
            lz.log.info(f"Load model from {FLAGS.ckpt.models}")
        if FLAGS.ckpt.buf:
            with open(FLAGS.ckpt.buf, 'rb') as f:
                buf = pickle.load(f)
                self.buf_real.add_transitions(buf)
            lz.log.warning(f"load model buffer from {FLAGS.ckpt.buf}")

    def save(self, fn):
        torch.save({
            "policy": self.policy.state_dict(),
            "h": self.h.state_dict(),
            "q1": self.qf1.state_dict(),
            "q2": self.qf2.state_dict(),
            "models": self.model.state_dict(),
        }, lz.log_dir / fn)

    def evaluate(self, tag: str, n_eval_samples: int):
        evaluate(self.n_samples_so_far, self.runners['evaluate'], self.mean_policy, f'evaluate/{tag}/mean_policy',
                 n_eval_samples=n_eval_samples)

    def explore(self, policy, n_samples):
        guarded_policy = ExplorationPolicy(policy, self.core_ref)

        self.n_samples_so_far += n_samples
        runner = self.runners['explore']
        core = self.core_ref

        tmp = rlz.ReplayBuffer(runner.envs[0], device=device, max_buf_size=n_samples)
        ep_infos = runner.run(guarded_policy, n_samples, buffer=tmp)
        self.buf_real += tmp

        for ep_info in ep_infos:
            self.n_expl_episodes += 1
            wandb.log({
                'explore': {
                    'return': ep_info['return'],
                    'is_safe': ep_info.get('episode.unsafe', 0),
                    'length': ep_info['length'],
                }
            })

        h = core.h(tmp['state'])
        u = core.u(tmp['state'], tmp['action'])
        nh = core.h(tmp['next_state'])

        nlls = core.model.get_nlls(tmp['state'], tmp['action'], tmp['next_state'])
        lz.log.debug(f'expl as val: {nlls}')

        n_model_failures = ((nh > 0) & (u <= 0)).sum()
        n_crabs_failures = ((h <= 0) & (u > 0)).sum()

        merged_infos = rlz.runner.merge_episode_stats(ep_infos)
        n_expl_unsafe_trajs = sum([info.get('episode.unsafe', 0) for info in ep_infos])
        max_h = h.max().item()
        lz.log.info(f"[explore] # {self.n_samples_so_far}: "
                    f"failure = [model = {n_model_failures}, crabs = {n_crabs_failures}], "
                    f"expl trajs return: {merged_infos['return']}, max h = {max_h:.6f}")

        if n_expl_unsafe_trajs > 0:
            lz.log.critical(f'[explore] {n_expl_unsafe_trajs} unsafe trajectories!')

        wandb.log({
            'explore.n_trajs': len(ep_infos),
            'explore.n_unsafe_trajs': n_expl_unsafe_trajs,
            'explore.n_model_failures': n_model_failures,
            'explore.n_crabs_failures': n_crabs_failures,
            'explore.policy_return': merged_infos['return'][0],
        })

        return tmp

    def _task_pretrain_h(self):
        self.h_opt.pretrain()
        self.save("init_h.pt")

    def train_models(self, *, epochs):
        self.model_trainer.max_epochs += epochs
        self.model_trainer.fit(
            self.model,
            train_dataloader=self.buf_real.data_loader(256, n_iters_per_epoch=1000, replace=True)
        )
        self.model.to(device)  # pytorch lightning transferred the model to cpu

    def train_models_2(self, *, epochs):
        pass

    def _task_train_policy(self):
        self.h_opt.h_ref = self.core_ref.h
        lz.log.info("pretrain s...")
        for i in range(FLAGS.n_pretrain_s_iters):
            if i % 1000 == 0:
                self.s_opt.debug(step=i)
            self.s_opt.step()

        # collect 10_000 samples
        self.evaluate("pre-run", 10_000)
        self.explore(self.policy, 10_000)
        self.train_models(epochs=5)

        # warming up Q
        self.policy_optimizer.can_update_policy = False
        for _ in range(10_001):
            self.policy_optimizer.step()
        self.policy_optimizer.can_update_policy = True

        freq = 0.5

        for epoch in range(101):
            if epoch % 10 == 0:
                self.save(f"ckpt-{epoch}.pt")
            self.evaluate("eval", 10_000)
            self.explore(rlz.policy.AddGaussianNoise(self.policy, 0.0, 2.0), self.horizon * 2)
            self.explore(rlz.policy.UniformPolicy(self.dim_action), self.horizon * 2)

            self.train_models(epochs=1)

            # train policy
            lz.log.info(f"Epoch {epoch}: train policy, safety req freq = {freq:.3f}")
            for _ in range(2_000):
                self.s_opt.step()
            for t in range(2_000):
                if t % 1000 == 0:
                    self.evaluate("eval", 10_000)
                    self.explore(self.policy, self.horizon * 1)

                if len(self.buf_real) > 1000:
                    self.policy_optimizer.step()   # optimize unsafe policy
                self.policy_adv_opt.step(freq)

            # train h
            lz.log.info(f"train h!")
            found, multiplier = self.h_opt.train()
            freq = np.clip(freq * multiplier, 0.1, 10)
            if found:
                lz.log.info(f"reduce frequency to {freq:.3f}. Reset core_ref")
                self.core_ref.load_state_dict(self.core.state_dict())
            else:
                lz.log.warning(f"can't find h, increase freq to {freq}")
                self.h.load_state_dict(self.core_ref.h.state_dict())
            self.h_opt.check_by_grad()


def main():
    import logging
    lz.init(Crabs.FLAGS)
    logging.getLogger('lightning').setLevel(0)
    torch.set_printoptions(linewidth=200)

    crabs = Crabs(make_env)
    crabs.run()


if __name__ == '__main__':
    main()
