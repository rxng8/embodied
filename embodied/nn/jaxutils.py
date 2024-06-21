import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj
from . import const

tfd = tfp.distributions
tfb = tfp.bijectors
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32


def cast_to_compute(values):
  return treemap(
      lambda x: x if x.dtype == const.COMPUTE_DTYPE else x.astype(const.COMPUTE_DTYPE),
      values)


def get_param_dtype():
  return const.PARAM_DTYPE


def check(predicate, message, **kwargs):
  if const.ENABLE_CHECKS:
    checkify.check(predicate, message, **kwargs)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


def scan(fun, carry, xs, unroll=False, axis=0):
  unroll = jax.tree_util.tree_leaves(xs)[0].shape[axis] if unroll else 1
  return nj.scan(fun, carry, xs, False, unroll, axis)


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
  tensor = tensor.astype(f32)  # To avoid overflows.
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).mean(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.seed(), values)[:amount]
  return values


def switch(pred, lhs, rhs):
  def fn(lhs, rhs):
    assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
    mask = pred
    while len(mask.shape) < len(lhs.shape):
      mask = mask[..., None]
    return jnp.where(mask, lhs, rhs)
  return treemap(fn, lhs, rhs)


def reset(xs, reset):
  def fn(x):
    mask = reset
    while len(mask.shape) < len(x.shape):
      mask = mask[..., None]
    return x * (1 - mask.astype(x.dtype))
  return treemap(fn, xs)


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(f32) > thres).astype(f32)
  neg = (target.astype(f32) <= thres).astype(f32)
  pred = (dist.mean().astype(f32) > thres).astype(f32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(f32).mean(),
      pred=dist.mean().astype(f32).mean(),
  )


def concat_dict(mapping, batch_shape=None):
  tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
  if batch_shape is not None:
    tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
  return jnp.concatenate(tensors, -1)


def onehot_dict(mapping, spaces, filter=False, limit=256):
  result = {}
  for key, value in mapping.items():
    if key not in spaces and filter:
      continue
    space = spaces[key]
    if space.discrete and space.dtype != jnp.uint8:
      if limit:
        assert space.classes <= limit, (key, space, limit)
      value = jax.nn.one_hot(value, space.classes)
    result[key] = value
  return result

