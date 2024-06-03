# %%

import embodied

from embodied.nn import SimpleEncoder

a = SimpleEncoder({}, act="hello", name="a")
b = SimpleEncoder({}, act="b", name="b")
SimpleEncoder.act

# %%

from embodied.api import make_logger
make_logger(embodied.Config(logdir="foo", filter=".*"))

# %%

import numpy as np
import jax
import jax.numpy as jnp
from embodied.nn import ResidualTimeBlock, ninjax as nj

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


# %%


import numpy as np
import jax
import jax.numpy as jnp
from embodied.nn import Attention, ninjax as nj

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
