
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
from .base import Initializer, Linear

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class TextEncoder(nj.Module):
  def __init__(self, vocab_size: int, outdim: int=128, time_dim: int=32, embed_dim: int=32, hidden: int=32) -> None:
    self._time_dim = time_dim
    self._vocab_size = vocab_size
    self._embed_dim = embed_dim
    self._hidden = hidden
    assert outdim % 2 == 0
    self._outdim = outdim // 2

  def __call__(self, tokens: jax.Array):
    # tokens: (*B, S) of type int
    *B, S = tokens.shape
    # encode position
    pos = jnp.arange(S)
    for _ in range(len(B)):
      pos = pos[None] # (*B, S)
    pos = jnp.tile(pos, (*B, 1)) # (..., 1, 1, S) => (*B, S)
    pos = jaxutils.cast_to_compute(pos)
    posinfo = self.get("sin", SinusoidalPositionEmbedding, self._time_dim)(pos) # (*B, S, time_dim)
    posinfo = self.get("posin", Linear, self._hidden)(posinfo)
    posinfo = jax.nn.gelu(posinfo)
    posinfo = self.get("posout", Linear, self._outdim)(posinfo)
    # encode text
    wordinfo = self.get("word", Embedding, self._vocab_size, self._embed_dim)(tokens)
    wordinfo = self.get("wordin", Linear, self._hidden)(wordinfo)
    wordinfo = jax.nn.gelu(wordinfo)
    wordinfo = self.get("wordout", Linear, self._outdim)(wordinfo)
    return jnp.concatenate([wordinfo, posinfo], axis=-1) # (*B, S, out_dim)