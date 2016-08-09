# -*- coding: utf-8 -*-

from masalachai.optimizer_scheduler import OptimizerScheduler


class DecayOptimizerScheduler(OptimizerScheduler):

    def __init__(self, optimizer, param_name, interval, decay_rate):
        super(DecayOptimizerScheduler, self).__init__(optimizer, param_name)
        self.interval = interval
        self.decay_rate = decay_rate

    def next(self, t):
        if (t % self.interval == 0) and (t != 0):
            self.param_value *= self.decay_rate
        return self.param_value
