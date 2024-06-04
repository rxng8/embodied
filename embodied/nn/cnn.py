
import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import functional
from . import distributions
from . import ninjax as nj
from .embeddings import SinusoidalPositionEmbedding, Embedding
from .base import Initializer, Linear, Conv2D, get_act, GroupNorm

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute

class ResidualBlock(nj.Module):

  act: str = "relu"
  norm: str = "instance"

  def __call__(self, x: jax.Array) -> jax.Array:
    # input: *B, H, W, C
    *B, H, W, C = x.shape
    res = functional.reflection_pad_2d(x, 1)
    res = self.get("res1", Conv2D, C, 3, stride=1,
      transp=False, act=self.act, norm=self.norm, pad='valid')(res)
    res = functional.reflection_pad_2d(res, 1)
    res = self.get("res2", Conv2D, C, 3, stride=1,
      transp=False, act='none', norm=self.norm, pad='valid')(res)
    return x + res


class ResidualTimeBlock(nj.Module):

  act: str = "silu"
  group: int = 1

  def __init__(self, dim: int) -> None:
    """Implement residual block with time integrated information as in LDM paper
      Resource: https://arxiv.org/pdf/2006.11239.pdf

    Args:
      dim (int): _description_
    """
    self._dim = dim
    self._act = get_act(self.act)

  def __call__(self, inputs: jax.Array, time_embed: jax.Array = None) -> jax.Array:
    x = self.get("conv", Conv2D, self._dim, 3, stride=1,
      transp=False, act='none', norm='none', pad='same')(inputs)
    x = self.get("convn", GroupNorm, self.group)(x)
    x = self._act(x)
    if time_embed is not None:
      t = self._act(time_embed) # (B, dim)
      t = self.get("time", Linear, 2 * self._dim)(t) # (B, 2dim)
      t = t[:, None, None, :] # (B, 1, 1, 2dim)
      shift, scale = jnp.split(t, 2, axis=-1) # (B, H, W, dim), (B, H, W, dim)
      x = x * (1 + scale) + shift # (B, H, W, dim)
    # cnn block
    x = self.get("conv2", Conv2D, self._dim, 3, stride=1,
      transp=False, act='none', norm='none', pad='same')(x)
    x = self.get("conv2n", GroupNorm, self.group)(x)
    x = self._act(x)
    # res
    res = self.get("res", Conv2D, self._dim, 1, stride=1,
      transp=False, act='none', norm='none', pad='same')(inputs)
    return x + res