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

