
import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from embodied.nn import ninjax as nj
from embodied.nn.cnn import ResidualBlock
import numpy as np


def test_residual_block():
  R = ResidualBlock(name="R")
  sample = jnp.asarray(np.random.normal(0, 1, size=(2, 8, 10, 16)))
  params = nj.init(R)({}, sample, seed=0)
  fn = jax.jit(nj.pure(R))
  _, out = fn(params, sample)
  assert out.shape == sample.shape