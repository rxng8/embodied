__version__ = '0.0.1'

from .core import *

from . import distr
from . import envs
from . import replay
from . import run
from . import nn

try:
  from rich import traceback
  import numpy as np
  import jax

  traceback.install(
      # show_locals=True,
      suppress=[np, jax])

except ImportError:
  pass
