import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj
from . import jaxutils

tfd = tfp.distributions
tfb = tfp.bijectors
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32
COMPUTE_DTYPE = f32
PARAM_DTYPE = f32
ENABLE_CHECKS = False



class Optimizer(nj.Module):

  # Normalization
  scaler: str = 'adam'
  eps: float = 1e-7
  beta1: float = 0.9
  beta2: float = 0.999

  # Learning rate
  warmup: int = 1000
  anneal: int = 0
  schedule: str = 'constant'

  # Regularization
  wd: float = 0.0
  wd_pattern: str = r'/kernel$'

  # Clipping
  pmin: float = 1e-3
  globclip: float = 0.0
  agc: float = 0.0

  # Smoothing
  momentum: bool = False
  nesterov: bool = False

  # Metrics
  details: bool = False

  def __init__(self, lr):
    self.lr = lr
    chain = []

    if self.globclip:
      chain.append(optax.clip_by_global_norm(self.globclip))
    if self.agc:
      chain.append(scale_by_agc(self.agc, self.pmin))

    if self.scaler == 'adam':
      chain.append(optax.scale_by_adam(self.beta1, self.beta2, self.eps))
    elif self.scaler == 'rms':
      chain.append(scale_by_rms(self.beta2, self.eps))
    else:
      raise NotImplementedError(self.scaler)

    if self.momentum:
      chain.append(scale_by_momentum(self.beta1, self.nesterov))

    if self.wd:
      assert not self.wd_pattern[0].isnumeric(), self.wd_pattern
      pattern = re.compile(self.wd_pattern)
      wdmaskfn = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(self.wd, wdmaskfn))

    if isinstance(self.lr, dict):
      chain.append(scale_by_groups({pfx: -lr for pfx, lr in self.lr.items()}))
    else:
      chain.append(optax.scale(-self.lr))

    self.chain = optax.chain(*chain)
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.chain = optax.apply_if_finite(
          self.chain, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')
    self.once = True

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    def wrapped(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    metrics = {}
    loss, params, grads, aux = nj.grad(
        wrapped, modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss /= self.grad_scale.read()
    if not isinstance(modules, (list, tuple)):
      modules = [modules]
    counts = {k: int(np.prod(v.shape)) for k, v in params.items()}
    if self.once:
      self.once = False
      prefs = []
      for key in counts:
        parts = key.split('/')
        prefs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), 2))]
      subcounts = {
          prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
          for prefix in set(prefs)}
      print(f'Optimizer {self.name} has {sum(counts.values()):,} variables:')
      for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
        print(f'{count:>14,} {prefix}')

    if jaxutils.parallel():
      grads = treemap(lambda x: jax.lax.pmean(x, 'i'), grads)
    if self.scaling:
      invscale = 1.0 / self.grad_scale.read()
      grads = treemap(lambda x: x * invscale, grads)
    optstate = self.get('state', self.chain.init, params)
    updates, optstate = self.chain.update(grads, optstate, params)
    self.put('state', optstate)

    if self.details:
      metrics.update(self._detailed_stats(optstate, params, updates, grads))

    scale = 1
    step = self.step.read().astype(f32)
    if self.warmup > 0:
      scale *= jnp.clip(step / self.warmup, 0, 1)
    assert self.schedule == 'constant' or self.anneal > self.warmup
    prog = jnp.clip((step - self.warmup) / (self.anneal - self.warmup), 0, 1)
    if self.schedule == 'constant':
      pass
    elif self.schedule == 'linear':
      scale *= 1 - prog
    elif self.schedule == 'cosine':
      scale *= 0.5 * (1 + jnp.cos(jnp.pi * prog))
    else:
      raise NotImplementedError(self.schedule)
    updates = treemap(lambda x: x * scale, updates)

    nj.context().update(optax.apply_updates(params, updates))
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm = optax.global_norm([x.find() for x in modules])
    isfin = jnp.isfinite
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = (~jnp.isfinite(grad_norm)).astype(f32)
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + isfin(grad_norm).astype(i32))
    else:
      jaxutils.check(isfin(grad_norm), f'{self.path} grad norm: {{x}}', x=grad_norm)
      self.step.write(self.step.read() + 1)
    jaxutils.check(isfin(update_norm), f'{self.path} updates: {{x}}', x=update_norm)
    jaxutils.check(isfin(param_norm), f'{self.path} params: {{x}}', x=param_norm)

    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = grad_norm
    metrics['update_norm'] = update_norm
    metrics['param_norm'] = param_norm
    metrics['grad_steps'] = self.step.read()
    metrics['param_count'] = jnp.array(sum(counts.values()))
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(i32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(f32) * self.grad_scale.read() +
        incr.astype(f32) * self.grad_scale.read() * 2 +
        decr.astype(f32) * self.grad_scale.read() / 2,
        1e-4, 1e5))
    return finite

  def _detailed_stats(self, optstate, params, updates, grads):
    groups = {
        'all': r'.*',
        'enc': r'/enc/.*',
        'dec': r'/dec/.*',
        'dyn': r'/dyn/.*',
        'con': r'/con/.*',
        'rew': r'/rew/.*',
        'actor': r'/actor/.*',
        'critic': r'/critic/.*',
        'out': r'/out/kernel$',
        'repr': r'/repr_logit/kernel$',
        'prior': r'/prior_logit/kernel$',
        'offset': r'/offset$',
        'scale': r'/scale$',
    }
    metrics = {}
    stddev = None
    for state in getattr(optstate, 'inner_state', optstate):
      if isinstance(state, optax.ScaleByAdamState):
        corr = 1 / (1 - 0.999 ** state.count)
        stddev = treemap(lambda x: jnp.sqrt(x * corr), state.nu)
    for name, pattern in groups.items():
      keys = [k for k in params if re.search(pattern, k)]
      ps = [params[k] for k in keys]
      us = [updates[k] for k in keys]
      gs = [grads[k] for k in keys]
      if not ps:
        continue
      metrics.update({f'{k}/{name}': v for k, v in dict(
          param_count=jnp.array(np.sum([np.prod(x.shape) for x in ps])),
          param_abs_max=jnp.stack([jnp.abs(x).max() for x in ps]).max(),
          param_abs_mean=jnp.stack([jnp.abs(x).mean() for x in ps]).mean(),
          param_norm=optax.global_norm(ps),
          update_abs_max=jnp.stack([jnp.abs(x).max() for x in us]).max(),
          update_abs_mean=jnp.stack([jnp.abs(x).mean() for x in us]).mean(),
          update_norm=optax.global_norm(us),
          grad_norm=optax.global_norm(gs),
      ).items()})
      if stddev is not None:
        sc = [stddev[k] for k in keys]
        pr = [
            jnp.abs(x) / jnp.maximum(1e-3, jnp.abs(y)) for x, y in zip(us, ps)]
        metrics.update({f'{k}/{name}': v for k, v in dict(
            scale_abs_max=jnp.stack([jnp.abs(x).max() for x in sc]).max(),
            scale_abs_min=jnp.stack([jnp.abs(x).min() for x in sc]).min(),
            scale_abs_mean=jnp.stack([jnp.abs(x).mean() for x in sc]).mean(),
            prop_max=jnp.stack([x.max() for x in pr]).max(),
            prop_min=jnp.stack([x.min() for x in pr]).min(),
            prop_mean=jnp.stack([x.mean() for x in pr]).mean(),
        ).items()})
    return metrics


def expand_groups(groups, keys):
  if isinstance(groups, (float, int)):
    return {key: groups for key in keys}
  groups = {
      group if group.endswith('/') else f'{group}/': value
      for group, value in groups.items()}
  assignment = {}
  groupcount = collections.defaultdict(int)
  for key in keys:
    matches = [prefix for prefix in groups if key.startswith(prefix)]
    if not matches:
      raise ValueError(
          f'Parameter {key} not fall into any of the groups:\n' +
          ''.join(f'- {group}\n' for group in groups.keys()))
    if len(matches) > 1:
      raise ValueError(
          f'Parameter {key} fall into more than one of the groups:\n' +
          ''.join(f'- {group}\n' for group in groups.keys()))
    assignment[key] = matches[0]
    groupcount[matches[0]] += 1
  for group in groups.keys():
    if not groupcount[group]:
      raise ValueError(
          f'Group {group} did not match any of the {len(keys)} keys.')
  expanded = {key: groups[assignment[key]] for key in keys}
  return expanded


def scale_by_groups(groups):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    scales = expand_groups(groups, updates.keys())
    updates = treemap(lambda u, s: u * s, updates, scales)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_agc(clip=0.03, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = treemap(fn, params, updates)
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = treemap(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = treemap(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = treemap(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = treemap(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)


class SlowUpdater(nj.Module):

  def __init__(self, src, dst, fraction=1.0, period=1):
    self.src = src
    self.dst = dst
    self.fraction = fraction
    self.period = period
    self.updates = nj.Variable(jnp.zeros, (), i32, name='updates')

  def __call__(self):
    assert self.src.find()
    updates = self.updates.read()
    need_init = (updates == 0).astype(f32)
    need_update = (updates % self.period == 0).astype(f32)
    mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
    params = {
        k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
        for k, v in self.src.find().items()}
    ema = treemap(
        lambda s, d: mix * s + (1 - mix) * d,
        params, self.dst.find())
    for name, param in ema.items():
      assert param.dtype == jnp.float32, (
          f'EMA of {name} should be float32 not {param.dtype}')
    self.dst.put(ema)
    self.updates.write(updates + 1)
