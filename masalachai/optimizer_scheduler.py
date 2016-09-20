# -*- coding: utf-8 -*-


class OptimizerScheduler(object):

    def __init__(self, optimizer, param_name):
        self.param_name = param_name
        self.param_value = optimizer.__dict__[self.param_name]

    def next(self, t):
        raise NotImplementedError
