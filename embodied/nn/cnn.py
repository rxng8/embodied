
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
from .base import Initializer, Linear, Conv2D

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
