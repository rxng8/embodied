
import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import functional
from . import distributions
from . import ninjax as nj
from .base import Initializer, Linear

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class Embedding(nj.Module):

  outscale: float = 1.0
  winit: str = 'normal'
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, count, units):
    self.count = count
    self.units = units
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)

  def __call__(self, x):
    assert x.dtype in (jnp.uint32, jnp.int32), x.dtype
    shape = (self.count, self.units)
    fan_shape = (1, self.units)
    w = self.get('embed', self._winit, shape, fan_shape)#.astype(x.dtype)
    return jnp.take(w, x, axis=0)


class SinusoidalPositionEmbedding(nj.Module):
  def __init__(self, dim: int) -> None:
    assert dim % 2 == 0, dim
    self._dim = dim
    self._d_model = dim // 2

  def __call__(self, pos: jax.Array):
    # pos: (*B): float32 -> out: (*B, dim): float32
    batch_dims = pos.shape
    i = jnp.arange(self._d_model)
    for _ in range(len(batch_dims)):
      i = i[None] # (*B, _d_model)
    pos_embedding = pos[..., None] * jnp.exp(-(2 * i / self._d_model) * jnp.log(10000))
    pos_embedding = jnp.concatenate([
      jnp.sin(pos_embedding),
      jnp.cos(pos_embedding)
    ], axis=-1)
    assert pos_embedding.shape == (*batch_dims, self._dim), pos_embedding.shape
    return pos_embedding


class TimeEmbedding(nj.Module):
  def __init__(self, dim: int):
    self._dim = dim # dimension of the sinusoidal
    self.sin_embed = SinusoidalPositionEmbedding(dim, name="sin_embed")

  def __call__(self, t: jax.Array):
    # t: (*B,): can be a jax integer, so cast to compute
    t = jaxutils.cast_to_compute(t)
    x = self.sin_embed(t) # (*B, dim)
    x = self.get("in", Linear, self._dim)(x) # (*B, dim)
    x = jax.nn.gelu(x) # (*B, dim)
    x = self.get("out", Linear, self._dim)(x) # (*B, dim)
    return x
