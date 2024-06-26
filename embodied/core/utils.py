import jax
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import sys

def timestamp(now=None, millis=False):
  now = datetime.now() if now is None else now
  string = now.strftime("%Y%m%dT%H%M%S")
  if millis:
    string += f'F{now.microsecond:06d}'
  return string


def check_vscode_interactive() -> bool:
  if hasattr(sys, 'ps1'):
    return True # ipython on Windows or WSL
  else: # check on linux: https://stackoverflow.com/a/39662359
    try:
      shell = get_ipython().__class__.__name__
      if shell == 'ZMQInteractiveShell':
        return True   # Jupyter notebook or qtconsole
      elif shell == 'TerminalInteractiveShell':
        return False  # Terminal running IPython
      else:
        return False  # Other type (?)
    except NameError:
      return False      # Probably standard Python interpreter

def tensorstats(tensor):
    """
    Prints tensor statistics (debugging tool).

    Args:
        tensor: argument tensor object to examine

    Returns:
        useful statistics to print to I/O
    """
    if isinstance(tensor, (np.ndarray, jax.Array, jnp.ndarray)):
        _tensor = np.asarray(tensor)
        return {
            'mean': _tensor.mean(),
            'std': _tensor.std(),
            'mag': np.abs(_tensor).max(),
            'min': _tensor.min(),
            'max': _tensor.max(),
        }
    elif isinstance(tensor, (list, tuple, dict)):
        try:
            values, _ = jax.tree.flatten(jax.tree.map(lambda x: x.flatten(), tensor))
            values = np.asarray(np.stack(values))
            return {
                'mean': values.mean(),
                'std': values.std(),
                'mag': np.abs(values).max(),
                'min': values.min(),
                'max': values.max(),
            }
        except:
            return None
    else:
        return None