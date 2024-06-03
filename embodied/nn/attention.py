
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
from .base import Initializer, Linear, GroupNorm, Norm

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute

class Attention(nj.Module):

  head: int = 1

  def __init__(self, hidden: int, **kwargs) -> None:
    """Usage: (B, T, E) -> (B, T, E)

    Args:
        hidden (int): _description_
        head (int): _description_
    """
    assert hidden % self.head == 0, f"hidden must be divisible by head, got hidden={hidden}, head={self.head}"
    # self._embed_dim = hidden # query dim * head: the dim of input
    self._hidden = hidden // self.head # dimension of the query: Q, we will project dim of key and value to this query dim
    self._kwargs = {**kwargs, 'act': 'none'} # make sure act is none

  def _cross_attention(self, query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
    """Implement cross attention mechanism

    References:
      - https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L4932
      * Sample weigths and bias when doing linear transformation of q, k, and v, basically, we project every tensor to q dim:
      ```python
        Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
        assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
        assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
        assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
        assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
        assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
        assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
      ```

    Eventually, Q == K == V.
    NOTE: key takeaway: the main pipeline/stream is the query running all the time, the conditioned input (infrequent) is the key and value
    NOTE: key takeaway: query is used to get the key => get weighted value.
    NOTE: according to latent diffusion model, the conditionin switchable is key and value, and the main running forward pass is query.
      This is also align with self attention where the main input is the query, and the key and value here is itself, the query.

    Args:
      query (jax.Array): (B, T, E): T is target sequence length, Q is the dimension of the query. Initially it's an arbitrary dim E => project to Q eventually
      key (jax.Array): (B, S, K): T is the source sequence length, K is the dimension of the key => project to Q eventually
      value (jax.Array): (B, S, V): S is the source seuqence length, V is the dimension of the value => project to Q eventually

    Returns:
      jax.Array: Attention output (B, T, E): T is the target sequence length, Q is the query dim => project back to original embedding dim E
    """
    # assert query.shape[-1] == self._embed_dim, ""
    B, T, E = query.shape #
    scale = 1.0 / jnp.sqrt(self._hidden) # ()
    query = self.get("q", Linear, (self._head, self._hidden), **self._kwargs)(query) # (B, T, H, Q)
    key = self.get("k", Linear, (self._head, self._hidden), **self._kwargs)(key) # (B, S, H, K) K==Q
    value = self.get("v", Linear, (self._head, self._hidden), **self._kwargs)(value) # (B, S, H, V) V==Q
    attention_weights = jnp.einsum("BTHQ,BSHQ->BTHS", query, key) # (B, T, H, S)
    attention_weights = attention_weights * scale # weighted scores
    attention_weights = jax.nn.softmax(attention_weights, axis=-1) # (B, T, H, S)
    attention_out = jnp.einsum("BTHS,BSHQ->BTHQ", attention_weights, value) # (B, T, H, Q)
    attention_out = attention_out.reshape((B, T, -1)) # (B, T, HQ)
    attention_out = self.get("out", Linear, E, **self._kwargs)(attention_out)
    return attention_out

  def self_attention(self, query):
    return self._cross_attention(query, query, query)

  def cross_attention(self, query, condition):
    return self._cross_attention(query, condition, condition)


class CrossAttentionBlock(nj.Module):

  head: int = 1
  group: int = 1

  def __init__(self, hidden: int, **kwargs) -> None:
    self._hidden = hidden
    self._kwargs = kwargs
    self.att = Attention(self._hidden, head=self.head, **kwargs)

  def __call__(self, query: jax.Array, condition: jax.Array) -> jax.Array:
    normq = self.get('normq', GroupNorm, self.group)(query)
    normc = self.get('normc', Norm, 'layer')(condition)
    att = self.att.cross_attention(normq, normc)
    return query + att

