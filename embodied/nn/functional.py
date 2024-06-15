from typing import List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from . import ninjax as nj
from .jaxutils import cast_to_compute, sg


# def bce(inputs: jax.Array, target: jax.Array, input_type='logit', eps=1e-5):
#   if input_type == 'logit':
#     return - (target * (jax.nn.log_sigmoid(inputs)))
#   elif input_type == 'probs':
#     return
#   else:
#     raise ValueError("`input_type` can only be `logit` or `probs`.")

def masked_fill(x: jax.Array, mask: jax.Array, other=0) -> jax.Array:
  """Return an output with masked condition, with non-masked value
    be the other value

  Args:
      x (jax.Array): _description_
      mask (jax.Array): _description_
      other (int, optional): _description_. Defaults to 0.

  Returns:
      jax.Array: _description_
  """
  return jnp.where(mask, x, jnp.broadcast_to(other, x.shape))

def reflection_pad_2d(x: jax.Array, pad: int):
  *B, H, W, C = x.shape
  pad_width = [(0, 0) for _ in range(len(B))] + [(pad, pad), (pad, pad), (0, 0)]
  return jnp.pad(x, pad_width, mode='reflect') # equals to reflection pad 2D

def bce(inputs: jax.Array, target: jax.Array, eps=1e-8):
  return - (target * (jnp.log(inputs.clip(eps))) + (1 - target) * jnp.log((1 - inputs).clip(eps)))


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def get_pixel_value(img: jax.Array, x: jax.Array, y: jax.Array):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    batch_size, height, width = x.shape
    batch_idx = jnp.arange(0, batch_size)
    batch_idx = batch_idx[..., None, None]
    b = jnp.tile(batch_idx, (1, height, width)) # (B, H, W)
    return img[b, y, x]


@jax.jit
def bilinear_sampler(img, grid, padding=0):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates (-1 to 1) provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout. Ranged from 0 to 1
    - grid: x, y which is the output of affine_grid_generator.
      (B, H_out, W_out, 2) in the form of x:  (B, H_out, W_out, 1)
      and y: (B, H_out, W_out, 1) => [x, y]@3, not [y, x]
    - padding: can be 0 or 1. The value to pad in the sampled image (which the
      grid is out of bound for the image)

    Returns
    -------
    - out: interpolated images according to grids.
      Same size as grid. (B, H_out, W_out, C).
      Also ranged from 0 to 1
    """
    B, H, W, C = img.shape
    x = grid[..., 0]
    y = grid[..., 1]

    max_y = jnp.astype(H - 1, 'int32')
    max_x = jnp.astype(W - 1, 'int32')
    zero = jnp.zeros((), dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = cast_to_compute(x)
    y = cast_to_compute(y)
    x = 0.5 * ((x + 1.0) * cast_to_compute(max_x-1))
    y = 0.5 * ((y + 1.0) * cast_to_compute(max_y-1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = jnp.astype(jnp.floor(x), 'int32')
    x1 = x0 + 1
    y0 = jnp.astype(jnp.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = jnp.clip(x0, zero, max_x)
    x1 = jnp.clip(x1, zero, max_x)
    y0 = jnp.clip(y0, zero, max_y)
    y1 = jnp.clip(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = cast_to_compute(x0)
    x1 = cast_to_compute(x1)
    y0 = cast_to_compute(y0)
    y1 = cast_to_compute(y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    # Create a mask to identify the elements in the grid that were out of bounds
    mask = ((x < 0) | (x >= W-1) | (y < 0) | (y >= H-1)).astype(jnp.float32)[..., None]

    # compute output
    return mask * padding + wa *Ia + wb*Ib + wc*Ic + wd*Id


class TPSGridGen(nj.Module):
  """Implement TPS Warping method based on the original TPS Warping paper[1]

    References:
      [1] F. L. Bookstein. Principal warps: Thin-plate splines and the
        decomposition of deformations. IEEE PAMI, 1989.
      [2] CP-VTON research work: https://github.com/sergeywong/cp-vton
  """
  def __init__(self, num_control_points: int, image_size: Tuple[int, int]) -> None:
    self.N = num_control_points
    self.image_size = image_size
    self.prepare()

  def initialize_tps(self, num_control_points: int, image_size: Tuple[int, int]) -> Tuple[np.ndarray]:
    """

    Args:
        num_control_points (int): _description_
        image_size (Tuple[int, int]): _description_

    Returns:
        L_inv: (1, N + 3, N + 3)
        control_point_x: (N,)
        control_point_y: (N,)
        img_grid_x: (1, H, W)
        img_grid_y: (1, H, W)
    """
    width, height = image_size
    img_grid_x, img_grid_y = np.meshgrid(
      np.linspace(-1, 1, width),
      np.linspace(-1, 1, height)
    ) # (H, W), (H, W)
    img_grid_x = img_grid_x[None] # (1, H, W, 1) => (B, H, W, 1)
    img_grid_y = img_grid_y[None] # (1, H, W, 1) => (B, H, W, 1)

    # Create source grid, num_control_points has to be square-root-able
    tps_grid_size = np.sqrt(num_control_points)
    assert tps_grid_size % 1 == 0, "num_control_points must be square-root-able"
    tps_axis_coords = np.linspace(-1, 1, int(tps_grid_size))
    # x and y are the x and y coordinates of each control point
    control_point_x, control_point_y = np.meshgrid(tps_axis_coords, tps_axis_coords) # ()
    control_point_x = control_point_x.flatten() # (N,)
    control_point_y = control_point_y.flatten() # (N,)
    x = control_point_x[:, None] # (N, 1)
    y = control_point_y[:, None] # (N, 1)

    # (N, 3)
    P = np.concatenate([
      np.ones((num_control_points, 1)),
      x,
      y
    ], axis=1)

    x_mat = np.repeat(x, num_control_points, 1) # (N, 1) => (N, N)
    y_mat = np.repeat(y, num_control_points, 1) # (N, 1) => (N, N)

    # compute all pair distances between each point x, y in N control points
    dist_mat = (x_mat - x_mat.transpose(1, 0))**2 + (y_mat - y_mat.transpose(1, 0))**2
    dist_mat[dist_mat == 0] = 1 # make diagonal 1 to avoid NaN in log computation

    # Compute K using the U function
    K = dist_mat * np.log(dist_mat) # (N, N)

    # construct matrix L
    L = np.concatenate([
      # concate the two top matrix
      np.concatenate([K, P], axis=1),
      # concate the two bottom matrices in col direction
      np.concatenate([P.transpose(1, 0), np.zeros((3, 3))], axis=1)
    ], axis=0)

    L_inv = np.linalg.inv(L)
    return L_inv[None], control_point_x, control_point_y, img_grid_x, img_grid_y

  def prepare(self):
    height = self.image_size[1]
    width = self.image_size[0]
    N = self.N

    self.L_inv, self.control_point_x, self.control_point_y,\
      img_grid_x, img_grid_y = self.initialize_tps(N, self.image_size)

    # repeat pre-defined control points along spatial dimensions of points to be transformed
    P_X = self.control_point_x[None, None, None, None] # (1, 1, 1, 1, N)
    P_X = np.repeat(P_X, height, 1) # (1, H, 1, 1, N)
    P_X = np.repeat(P_X, width, 2) # (1, H, W, 1, N)
    P_Y = self.control_point_y[None, None, None, None] # (1, 1, 1, 1, N)
    P_Y = np.repeat(P_Y, height, 1) # (1, H, 1, 1, N)
    P_Y = np.repeat(P_Y, width, 2) # (1, H, W, 1, N)

    # compute distance P_i - (grid_X, grid_Y)
    # grid is expanded in point dim 4, but not in batch dim 0, as points P_X, P_Y are fixed for all batch
    # (1, H, W, 1, N)
    points_X_for_summation = img_grid_x[..., None, None].repeat(N, -1)
    points_Y_for_summation = img_grid_y[..., None, None].repeat(N, -1)

    # use expanded P_X,P_Y in batch dimension
    # (1, H, W, 1, N) - (1, H, W, 1, N) = (1, H, W, 1, N)
    delta_X = points_X_for_summation - P_X
    delta_Y = points_Y_for_summation - P_Y

    dist_squared = delta_X**2 + delta_Y**2
    dist_squared[dist_squared == 0] = 1 # avoid NaN in log computation
    self.U = dist_squared * np.log(dist_squared) # (1, H, W, 1, N)

    # expand grid in batch dimension if necessary
    self.points_X_batch = img_grid_x[..., None] # (1, H, W, 1)
    self.points_Y_batch = img_grid_y[..., None] # (1, H, W, 1)

  def __call__(self, theta: jax.Array) -> jax.Array:
    """Return a Thin-Plate-Spline Transformation given theta

    Args:
      theta (jax.Array): (B, 2 * N). N is the number of control points.
        Must be square-root-able.

    Returns:
      jax.Array: (B, H, W, 2). x and y grid coordinates of the TPS transformation.
        Normalized coordinates ranged from -1 to 1
    """
    height = self.image_size[1]
    width = self.image_size[0]
    N = self.N
    B, _ = theta.shape

    # JAX computation starts from here
    control_point_x = cast_to_compute(self.control_point_x)
    control_point_y = cast_to_compute(self.control_point_y)
    U = cast_to_compute(self.U)
    L_inv = cast_to_compute(self.L_inv)
    points_X_batch = cast_to_compute(self.points_X_batch)
    points_Y_batch = cast_to_compute(self.points_Y_batch)

    # split theta into point coordinates
    Q_X, Q_Y = jnp.split(theta, 2, axis=-1) # (B, N), (B, N)

    Q_X = Q_X[..., None] + jnp.asarray(control_point_x)[None, ..., None] # (B, N, 1)
    Q_Y = Q_Y[..., None] + jnp.asarray(control_point_y)[None, ..., None] # (B, N, 1)

    # compute weigths for non-linear part: (B, N, N) @ (B, N, 1) = (B, N, 1)
    W_X = jnp.einsum("bij,bjk->bik", jnp.repeat(L_inv[:, :N, :N], B, axis=0), Q_X)
    W_Y = jnp.einsum("bij,bjk->bik", jnp.repeat(L_inv[:, :N, :N], B, axis=0), Q_Y)

    # reshape W_X and W_Y: (B, N, 1) => (B, H, W, 1, N)
    W_X = jnp.transpose(W_X, [0, 2, 1]) # (B, N, 1) => (B, 1, N)
    W_X = W_X[:, None, None] # (B, 1, N) => (B, 1, 1, 1, N)
    W_X = jnp.tile(W_X, (1, height, width, 1, 1)) # (B, H, W, 1, N)
    W_Y = jnp.transpose(W_Y, [0, 2, 1]) # (B, N, 1) => (B, 1, N)
    W_Y = W_Y[:, None, None] # (B, 1, N) => (B, 1, 1, 1, N)
    W_Y = jnp.tile(W_Y, (1, height, width, 1, 1)) # (B, H, W, 1, N)

    # compute weights for affine part: (B, 3, N) @ (B, N, 1) = (B, N, 1)
    A_X = jnp.einsum("bij,bjk->bik", jnp.repeat(L_inv[:, N:, :N], B, axis=0), Q_X)
    A_Y = jnp.einsum("bij,bjk->bik", jnp.repeat(L_inv[:, N:, :N], B, axis=0), Q_Y)

    # reshape A_X and A_Y: (B, N, 1) => (B, H, W, 1, N)
    A_X = jnp.transpose(A_X, [0, 2, 1]) # (B, N, 1) => (B, 1, N)
    A_X = A_X[:, None, None] # (B, 1, N) => (B, 1, 1, 1, N)
    A_X = jnp.tile(A_X, (1, height, width, 1, 1)) # (B, H, W, 1, N)
    A_Y = jnp.transpose(A_Y, [0, 2, 1]) # (B, N, 1) => (B, 1, N)
    A_Y = A_Y[:, None, None] # (B, 1, N) => (B, 1, 1, 1, N)
    A_Y = jnp.tile(A_Y, (1, height, width, 1, 1)) # (B, H, W, 1, N)

    # (B, H, W, 1)
    points_X_prime = A_X[..., 0] + \
      A_X[..., 1] * points_X_batch + \
      A_X[..., 2] * points_Y_batch + \
      (W_X * U).sum(-1)

    # (B, H, W, 1)
    points_Y_prime = A_Y[..., 0] + \
      A_Y[..., 1] * points_X_batch + \
      A_Y[..., 2] * points_Y_batch + \
      (W_Y * U).sum(-1)

    # (B, H, W, 1)
    return jnp.concatenate([points_X_prime,points_Y_prime], -1)


class AffineGridGen(nj.Module):
  """Implement TPS Warping method based on the original TPS Warping paper[1]

    References:
      [1] F. L. Bookstein. Principal warps: Thin-plate splines and the
        decomposition of deformations. IEEE PAMI, 1989.
      [2] CP-VTON research work: https://github.com/sergeywong/cp-vton
  """
  pass


def l2_norm(x: jax.Array):
  dtype = x.dtype
  L = x.shape[-1]
  epsilon = 1e-6
  x = jnp.sqrt((x**2).sum(-1, keepdims=True) + epsilon).repeat(L, -1) # (*B, 1) -> (*B, L)
  return x.astype(dtype)

def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + jnp.tanh(sqrt_2_over_pi * (x + coeff * jnp.power(x, 3))))

def correlation(x1: jax.Array, x2: jax.Array):
  B, H, W, C = x1.shape
  x1 = x1.transpose([0, 3, 2, 1]) # (B, C, W, H)
  x1 = x1.reshape([B, C, W*H]) # (B, C, WH)

  x2 = x2.reshape([B, C, H*W])
  x2 = x2.transpose([0, 2, 1]) # (B, HW, C)

  corr = jnp.einsum("bij,bjk->bik", x2, x1) # (B, HW, WH)
  corr = corr.reshape([B, H, W, W*H]) # (B, H, W, WH)
  return corr
