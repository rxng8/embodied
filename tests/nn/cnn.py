
import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from embodied.nn import ninjax as nj
from embodied.nn.cnn import ResidualBlock, ResidualTimeBlock
import numpy as np


def test_residual_block():
  R = ResidualBlock(name="R")
  sample = jnp.asarray(np.random.normal(0, 1, size=(2, 8, 10, 16)))
  params = nj.init(R)({}, sample, seed=0)
  fn = jax.jit(nj.pure(R))
  _, out = fn(params, sample)
  assert out.shape == sample.shape

def test_residual_block_2():
  R = ResidualBlock(name="R")
  sample = jnp.asarray(np.random.normal(0, 1, (8, 16, 16, 256)))
  params = nj.init(R)({}, sample, seed=0)
  fn = jax.jit(nj.pure(R))
  _, out = fn(params, sample)
  assert sample.shape == out.shape

def test_residual_time_block():
  out_dim = 8
  R = ResidualTimeBlock(out_dim, group=2, name="R")
  B, H, W, C = 1, 64, 64, 3
  Z = 128
  sample_img = jnp.asarray(np.random.normal(0,1,(B, H, W, C)))
  sample_timemb = jnp.asarray(np.random.normal(0, 1, (B, Z)))
  params = nj.init(R)({}, sample_img, sample_timemb, seed=0)
  forward = jax.jit(nj.pure(R))
  _, out = forward(params, sample_img, sample_timemb)
  assert out.shape == (B, H, W, out_dim)