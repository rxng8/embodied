import jax
import numpy as np
import jax.numpy as jnp

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