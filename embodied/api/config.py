
import pathlib
import embodied
from embodied import Config
from ruamel import yaml
from embodied import Flags

def load_config(config_path: pathlib.Path | embodied.Path, argv=[]) -> Config:
  """Accept a YAML file config with file structures:
    ```
    defaults:
      logroot: path/to/your/logroot
      expname: name_of_your_experiment
      logdir: ""
      # other configs
    # other mergeable config
    ```
    and create logdir, save the processed config.yaml file inside
      the logdir

    Other file structure without logroot and logdir can also be used, this will not
      create a folder and will not save the config inside the logdir

  Args:
      config_path (str | pathlib.Path | embodied.Path): _description_
      argv (list, optional): _description_. Defaults to [].

  Returns:
      Config: _description_
  """
  assert isinstance(config_path, (pathlib.Path, embodied.Path)), "config path should be in pathlib.Path or embodied.Path format"
  configs = yaml.YAML(typ='safe').load(config_path)
  parsed, other = Flags(configs=['defaults']).parse_known(argv)
  # Preping and parsing all configs and overrides
  config = Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = Flags(config).parse(other)
  # logdir initialization: logdir = logroot / expname
  if "logroot" in config and "expname" in config and "logdir" in config and config.logroot and config.expname and config.logdir == "":
    logdir = embodied.Path(config.logroot) / config.expname
    logdir.mkdir()
    config = config.update({"logdir": str(logdir)})
    # Save the config inside logdir
    config.save(logdir / 'config.yaml')
  # Or simply take the logdir if the logdir is available
  elif "logdir" in config and config.logdir:
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    # Save the config inside logdir
    config.save(logdir / 'config.yaml')
  return config