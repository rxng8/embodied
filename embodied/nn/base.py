import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import functional
from . import distributions
from . import ninjax as nj
from . import const

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class SimpleEncoder(nj.Module):

  depth: int = 128
  mults: tuple = (1, 2, 4, 2)
  layers: int = 5
  units: int = 1024
  symlog: bool = True
  norm: str = 'rms'
  act: str = 'gelu'
  kernel: int = 4
  outer: bool = False
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.spaces = spaces
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.vecinp = Input(self.veckeys, featdims=1)
    self.imginp = Input(self.imgkeys, featdims=3)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, data, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = []

    shape = data['is_first'].shape[:bdims]
    data = {k: data[k] for k in self.spaces}
    data = jaxutils.onehot_dict(data, self.spaces)

    if self.veckeys:
      x = self.vecinp(data, bdims, f32)
      x = x.reshape((-1, *x.shape[bdims:]))
      x = jaxutils.symlog(x) if self.symlog else x
      x = jaxutils.cast_to_compute(x)
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      outs.append(x)

    if self.imgkeys:
      print('ENC')
      x = self.imginp(data, bdims, const.COMPUTE_DTYPE) - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        stride = 1 if self.outer and i == 0 else 2
        x = self.get(f'conv{i}', Conv2D, depth, self.kernel, stride, **kw)(x)
      assert x.shape[-3] == x.shape[-2] == self.minres, x.shape
      x = x.reshape((x.shape[0], -1))
      print(x.shape, 'out')
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    x = x.reshape((*shape, *x.shape[1:]))
    return x


class SimpleDecoder(nj.Module):

  inputs: tuple = ('deter', 'stoch')
  depth: int = 128
  mults: tuple = (1, 2, 4, 3)
  sigmoid: bool = True
  layers: int = 5
  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  vecdist: str = 'symlog_mse'
  kernel: int = 4
  outer: bool = False
  block_fans: bool = False
  block_norm: bool = False
  block_space: int = 0
  hidden_stoch: bool = False
  space_hidden: int = 0
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.inp = Input(self.inputs, featdims=1)
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.spaces = spaces
    self.depths = tuple([self.depth * mult for mult in self.mults])
    self.imgdep = sum(self.spaces[k].shape[-1] for k in self.imgkeys)
    self.kw = kw

  def __call__(self, lat, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = {}

    if self.veckeys:
      inp = self.inp(lat, bdims, const.COMPUTE_DTYPE)
      x = inp.reshape((-1, inp.shape[-1]))
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      x = x.reshape((*inp.shape[:bdims], *x.shape[1:]))
      for k in self.veckeys:
        dist = (
            dict(dist='softmax', bins=self.spaces[k].classes)
            if self.spaces[k].discrete else dict(dist=self.vecdist))
        k = k.replace('/', '_')
        outs[k] = self.get(f'out_{k}', Dist, self.spaces[k].shape, **dist)(x)

    if self.imgkeys:
      inp = self.inp(lat, bdims, const.COMPUTE_DTYPE)
      print('DEC')
      shape = (self.minres, self.minres, self.depths[-1])
      x = inp.reshape((-1, inp.shape[-1]))

      if self.space_hidden:
        x = self.get('space0', Linear, self.space_hidden * self.units, **kw)(x)
        x = self.get('space1', Linear, shape, **kw)(x)
      elif self.block_space:
        g = self.block_space
        x0 = einops.rearrange(cast(lat['deter']), 'b t ... -> (b t) ...')
        x1 = einops.rearrange(cast(lat['stoch']), 'b t l c -> (b t) (l c)')
        x0 = self.get(
            'space0', BlockLinear, int(np.prod(shape)), g, **self.kw,
            block_fans=self.block_fans, block_norm=self.block_norm)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=self.minres, w=self.minres, g=g)
        if self.hidden_stoch:
          x1 = self.get('space1hid', Linear, 2 * self.units, **kw)(x1)
        x1 = self.get('space1', Linear, shape, **self.kw)(x1)
        x = self.get('spacenorm', Norm, self.norm, act=self.act)(x0 + x1)
      else:
        x = self.get('space', Linear, shape, **kw)(x)

      print(x.shape, 'in')
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        x = self.get(
            f'conv{i}', Conv2D, depth, self.kernel, 2, **kw, transp=True)(x)
      outkw = dict(**self.kw, outscale=self.outscale, transp=True)
      stride = 1 if self.outer else 2
      x = self.get(
          'imgout', Conv2D, self.imgdep, self.kernel, stride, **outkw)(x)
      x = jax.nn.sigmoid(x) if self.sigmoid else x + 0.5
      print(x.shape, 'out')
      x = x.reshape((*inp.shape[:bdims], *x.shape[1:]))
      split = np.cumsum([self.spaces[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        outs[k] = jaxutils.MSEDist(f32(out), 3, 'sum')

    return outs


class MLP(nj.Module):

  layers: int = None
  units: int = None
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, shape, dist='mse', inputs=['tensor'], **kw):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
    assert isinstance(shape, (tuple, dict, type(None))), shape
    assert isinstance(dist, (str, dict)), dist
    assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)
    self.shape = shape
    self.dist = dist
    self.inputs = Input(inputs, featdims=1)
    distonly = ('outscale', 'minstd', 'maxstd', 'unimix', 'bins')
    self.lkw = {k: v for k, v in kw.items() if k not in distonly}
    forbidden = ('binit', 'norm', 'act')
    self.dkw = {k: v for k, v in kw.items() if k not in forbidden}

  def __call__(self, inputs, bdims=2, training=False):
    feat = self.inputs(inputs, bdims, const.COMPUTE_DTYPE)
    x = feat.reshape([-1, feat.shape[-1]])
    for i in range(self.layers):
      x = self.get(f'h{i}', Linear, self.units, **self.lkw)(x)
    x = x.reshape((*feat.shape[:bdims], -1))
    if self.shape is None:
      return x
    elif isinstance(self.shape, dict):
      return {
          k: self._out(k, v, self.dist[k], x) for k, v in self.shape.items()}
    else:
      return self._out('dist', self.shape, self.dist, x)

  def _out(self, name, shape, dist, x):
    name = name.replace('/', '_').replace('.', '_')
    return self.get(name, Dist, shape, dist, **self.dkw)(x)


class Dist(nj.Module):

  outscale: float = 0.1
  minstd: float = 1.0
  maxstd: float = 1.0
  unimix: float = 0.0
  bins: int = 255

  def __init__(self, shape, dist='mse', **kw):
    assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
    forbidden = ('binit', 'norm', 'act')
    assert all(k not in kw for k in forbidden), (forbidden, kw)
    self.shape = shape
    self.dist = dist
    self.kw = dict(**kw, outscale=self.outscale)

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    shape = self.shape
    padding = 0

    if 'twohot' in self.dist or self.dist == 'softmax':
      padding = int(self.bins % 2)
      shape = (*self.shape, self.bins + padding)

    out = self.get('out', Linear, int(np.prod(shape)), **self.kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    out = out[..., :-padding] if padding else out

    if 'normal' in self.dist:
      units = int(np.prod(self.shape))
      std = self.get('std', Linear, units, **self.kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self.shape).astype(f32)

    if self.dist == 'symlog_mse':
      fwd, bwd = jaxutils.symlog, jaxutils.symexp
      return distributions.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'hyperbolic_mse':
      fwd = lambda x, eps=1e-3: (
          jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x)
      bwd = lambda x, eps=1e-3: jnp.sign(x) * (jnp.square(
          jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps -
          1 / 2 / eps) - 1)
      return distributions.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'symlog_and_twohot':
      bins = np.linspace(-20, 20, out.shape[-1])
      return distributions.TwoHotDist(
          out, bins, len(self.shape), functional.symlog, functional.symexp)

    if self.dist == 'symexp_twohot':
      if out.shape[-1] % 2 == 1:
        half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
        half = functional.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
      else:
        half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], 0)
      return distributions.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'hyperbolic_twohot':
      eps = 0.001
      f = lambda x: np.sign(x) * (np.square(np.sqrt(
          1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps) - 1)
      bins = f(np.linspace(-300, 300, out.shape[-1]))
      return distributions.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'mse':
      return distributions.MSEDist(out, len(self.shape), 'sum')

    if self.dist == 'huber':
      return distributions.HuberDist(out, len(self.shape), 'sum')

    if self.dist == 'normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self.shape) * tfd.Normal(0.0, hi).entropy()
      return dist

    if self.dist == 'trunc_normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * (
          tfd.TruncatedNormal(1.0, lo, -1, 1).entropy())
      dist.maxent = np.prod(self.shape) * (
          tfd.TruncatedNormal(0.0, hi, -1, 1).entropy())
      return dist

    if self.dist == 'binary':
      dist = tfd.Bernoulli(out)
      if self.shape:
        dist = tfd.Independent(dist, len(self.shape))
      return dist

    if self.dist == 'softmax':
      dist = tfd.Categorical(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      return dist

    if self.dist == 'onehot':
      if self.unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        out = jnp.log(probs)
      dist = distributions.OneHotDist(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self.shape[:-1]) * np.log(self.shape[-1])
      return dist

    raise NotImplementedError(self.dist)


class Conv2D(nj.Module):

  groups: int = 1
  transp: bool = False
  act: str = 'none'
  norm: str = 'none'
  pad: str = 'same'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = kernel
    self.stride = stride
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (self.kernel, self.kernel, x.shape[-1], self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
      x = jax.lax.conv_transpose(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    else:
      G = self.groups
      shape = (self.kernel, self.kernel, x.shape[-1] // G, self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
    if self.bias:
      if self.binit:
        args = (self._winit, self.depth, shape)
      else:
        args = (self._binit, self.depth)
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(x.shape[-3:]))
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Linear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  fanin: int = 0

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    shape = (x.shape[-1], int(np.prod(self.units)))
    fan_shape = (self.fanin, shape[1]) if self.fanin else None
    x = x @ self.get('kernel', self._winit, shape, fan_shape).astype(x.dtype)
    flops = int(np.prod(shape))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(self.units))
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    return x


class BlockLinear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, units, groups):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    assert groups <= np.prod(units), (groups, units)
    self.groups = groups
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype,
        block_fans=self.block_fans)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    if self.block_norm:
      self._norm = [
          Norm(self.norm, name=f'norm{i}') for i in range(self.groups)]
    else:
      self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    if self.block_norm and self._norm != 'none':
      x = jnp.concatenate([
          f(y) for f, y in zip(self._norm, jnp.split(x, self.groups, -1))], -1)
    else:
      x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    bdims, indim, outdim = x.shape[:-1], x.shape[-1], np.prod(self.units)
    if indim % self.groups != 0:
      pad = int(np.ceil(indim / self.groups)) * self.groups - indim
      x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], pad), x.dtype)], -1)
      indim = x.shape[-1]
    assert indim % self.groups == outdim % self.groups == 0, (
        indim, outdim, self.groups, self.units)
    shape = (self.groups, indim // self.groups, outdim // self.groups)
    kernel = self.get('kernel', self._winit, shape, shape).astype(x.dtype)
    flops = int(np.prod(shape))
    x = x.reshape((*bdims, self.groups, indim // self.groups))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*bdims, outdim))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      bias = self.get('bias', *args)
      x += bias.astype(x.dtype)
      flops += int(np.prod(self.units))
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    assert x.dtype == const.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Norm(nj.Module):

  act: str = 'none'

  def __init__(self, impl, eps=1e-4):
    if '1em' in impl:
      impl, exponent = impl.split('1em')
      eps = 10 ** -int(exponent)
    self._impl = impl
    self._eps = eps

  def __call__(self, x):
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _norm(self, x):
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      mean = x.mean(-1)[..., None]
      # mean2 = jnp.square(x).mean(-1)[..., None]
      # var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = ((x - mean)**2).mean(-1)[..., None]
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      mult = scale * jax.lax.rsqrt(var + self._eps)
      x = (x - mean) * mult + offset
      return cast(x)
    elif self._impl == 'rms':
      dtype = x.dtype
      x = f32(x) if x.dtype == jnp.float16 else x
      scale = self.get('scale', jnp.ones, x.shape[-1], f32).astype(x.dtype)
      mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * scale
      return (x * mult).astype(dtype)
    elif self._impl == 'rms_instance':
      x = x.astype(f32)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
      mult = mult * scale
      return cast(x * mult)
    elif self._impl == 'grn':
      assert len(x.shape) >= 4, x.shape
      x = x.astype(f32)
      norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
      norm /= (norm.mean(-1, keepdims=True) + self._eps)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (norm * scale + 1) * x + offset
      return cast(x)
    elif self._impl == 'instance':
      x = x.astype(f32)
      mean = x.mean(axis=(-3, -2), keepdims=True)
      var = x.var(axis=(-3, -2), keepdims=True)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
      return cast(x)
    else:
      raise NotImplementedError(self._impl)


class GroupNorm(nj.Module):

  def __init__(self, groups: int):
    self._groups = groups

  def __call__(self, x):
    dtype = x.dtype
    x = x.astype(f32)
    x = x.reshape((*x.shape[:-1], self._groups, -1))
    x = jax.nn.standardize(x, axis=-1, epsilon=1e-5)
    x *= self.get('scale', jnp.ones, x.shape[-1], f32)
    x += self.get('bias', jnp.zeros, x.shape[-1], f32)
    x = x.reshape((*x.shape[:-2], -1))
    x = x.astype(dtype)
    return x


class Moments(nj.Module):

  rate: float = 0.01
  limit: float = 1e-8
  perclo: float = 5.0
  perchi: float = 95.0

  def __init__(self, impl='mean_std'):
    self.impl = impl
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    elif self.impl == 'min_max':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc_corr':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update=True):
    update and self.update(x)
    return self.stats()

  def update(self, x):
    if jaxutils.parallel():
      mean = lambda x: jax.lax.pmean(x.mean(), 'i')
      min_ = lambda x: jax.lax.pmin(x.min(), 'i')
      max_ = lambda x: jax.lax.pmax(x.max(), 'i')
      per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, 'i'), q)
    else:
      mean = jnp.mean
      min_ = jnp.min
      max_ = jnp.max
      per = jnp.percentile
    x = sg(x.astype(f32))
    m = self.rate
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean.write((1 - m) * self.mean.read() + m * mean(x))
      self.sqrs.write((1 - m) * self.sqrs.read() + m * mean(x * x))
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    elif self.impl == 'min_max':
      low, high = min_(x), max_(x)
      self.low.write((1 - m) * jnp.minimum(self.low.read(), low) + m * low)
      self.high.write((1 - m) * jnp.maximum(self.high.read(), high) + m * high)
    elif self.impl == 'perc':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
    elif self.impl == 'perc_corr':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    else:
      raise NotImplementedError(self.impl)

  def stats(self):
    if self.impl == 'off':
      return 0.0, 1.0
    elif self.impl == 'mean_std':
      corr = jnp.maximum(self.rate, self.corr.read())
      mean = self.mean.read() / corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() / corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return sg(mean), sg(std)
    elif self.impl == 'min_max':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc_corr':
      corr = jnp.maximum(self.rate, self.corr.read())
      lo = self.low.read() / corr
      hi = self.high.read() / corr
      span = hi - lo
      span = jnp.maximum(self.limit, span)
      return sg(lo), sg(span)
    else:
      raise NotImplementedError(self.impl)


class Input:

  def __init__(self, keys=['tensor'], featdims=1):
    self.keys = tuple(keys)
    self.featdims = featdims

  def __call__(self, inputs, bdims=2, dtype=None):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    try:
      xs = []
      for key in self.keys:
        x = inputs[key]
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
          x = jnp.concatenate([x.real, x.imag], -1)
        x = x.astype(dtype or inputs[self.keys[0]].dtype)
        x = x.reshape((*x.shape[:bdims + self.featdims - 1], -1))
        msg = f'Invalid input ({nj.SCOPE}, {key}, {x.shape}, {x.dtype}): {{x}}'
        jaxutils.check(jnp.isfinite(x).all(), msg, x=x)
        xs.append(x)
      xs = jnp.concatenate(xs, -1)
    except (KeyError, ValueError, TypeError) as e:
      shapes = {k: v.shape for k, v in inputs.items()}
      raise ValueError(
          f'Error: {e}\n'
          f'Input shapes: {shapes}\n' +
          f'Requested keys: {self.keys}')
    return xs


class Initializer:

  VARIANCE_FACTOR = 1.0

  def __init__(
      self, dist='normal', scale=1.0, fan='in', dtype='default',
      block_fans=False):
    self.dist = dist
    self.scale = scale
    self.fan = fan
    self.dtype = dtype
    self.block_fans = block_fans

  def __call__(self, shape, fan_shape=None):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    assert all(x > 0 for x in shape), shape
    dtype = const.PARAM_DTYPE if self.dtype == 'default' else self.dtype
    dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype
    fanin, fanout = self._fans(fan_shape or shape)
    fan = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}[self.fan]
    if self.dist == 'zeros':
      value = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(self.VARIANCE_FACTOR / fan)
      value = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      value = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      value *= 1.1368 * np.sqrt(self.VARIANCE_FACTOR / fan)
      value = value.astype(dtype)
    elif self.dist == 'normed':
      value = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      value /= jnp.linalg.norm(value.reshape((-1, shape[-1])), 2, 0)
    elif self.dist == 'complex':
      assert jnp.issubdtype(dtype, jnp.complexfloating), dtype
      realdt = jnp.finfo(dtype).dtype
      value = jax.random.truncated_normal(
          nj.seed(), -2, 2, (2, *shape), realdt)
      value = value[0] + 1j * value[1]
      value *= jax.lax.convert_element_type(1.137 * np.sqrt(1 / fan), realdt)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.seed(), matshape, dtype)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    value *= self.scale
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    elif len(shape) == 3 and self.block_fans:
      return shape[1:]
    else:
      space = int(np.prod(shape[:-2]))
      return (shape[-2] * space, shape[-1] * space)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'gelu_tanh':
    return functional.gelu_tanh
  elif name == 'cswiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      y1, y2 = jnp.split(y, 2, -1)
      pad = jnp.ones_like(y1)
      x = jax.nn.swish(jnp.concatenate([x, -x], -1))
      y = jnp.concatenate([y1, pad, y2, pad], -1)
      return x * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
