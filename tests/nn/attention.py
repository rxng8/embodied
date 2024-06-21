
import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from embodied.nn import Attention, ninjax as nj


def test_cross_attention():
  hidden = 4
  A = Attention(hidden, name="A")
  B, H, W, C = 1, 64, 64, 256
  S, Z = 18, 128
  sample_img = jnp.asarray(np.random.normal(0,1,(B, H, W, C)))
  sample_cond = jnp.asarray(np.random.normal(0, 1, (B, S, S, S+3, Z)))
  params = nj.init(A.cross_attention)({}, sample_img, sample_cond, seed=0)
  forward = jax.jit(nj.pure(A.cross_attention))
  _, out = forward(params, sample_img, sample_cond)
  # assert out.shape == (B, H, W, out_dim)
  out.shape


