
import embodied

def make_logger(config):
  logdir = embodied.Path(config.logdir)
  return embodied.Logger(embodied.Counter(), [
    embodied.logger.TerminalOutput(config.filter),
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    embodied.logger.TensorBoardOutput(logdir),
    # embodied.logger.WandbOutput(logdir.name, config=config),
  ])