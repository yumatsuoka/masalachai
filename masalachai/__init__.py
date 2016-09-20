import pkg_resources

from masalachai import logger
from masalachai import loggers
from masalachai import datafeeder
from masalachai import datafeeders
from masalachai import optimizer_scheduler
from masalachai import optimizer_schedulers
from masalachai import trainer
from masalachai import trainers
from masalachai import model
from masalachai import models
from masalachai import preprocesses

__version__ = pkg_resources.get_distribution('masalachai').version

Logger = logger.Logger
DataFeeder = datafeeder.DataFeeder
OptimizerScheduler = optimizer_scheduler.OptimizerScheduler
Trainer = trainer.Trainer
Model = model.Model
