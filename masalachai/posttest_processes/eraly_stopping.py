# -*- coding: utf-8 -*-

import os
from masalachai.posttest_processes.snapshot_best import SnapshotBest
from chainer import serializers
from scipy.interpolate import InterpolatedUnivariateSpline

class EarlyStopping(SnapshotBest):

    def __init__(self, model, filename, observed='loss', patience=1, imporovement=0.995, interval=10):
        super(EarlyStopping, self).__init__(
                mode, filename, observed=observed, patience=patience)
        self.imporovement = imporovement
        self.interval = interval

    def __call__(self, test_res):
        value = test_res[self.observed]
        self.iteration = test_res['iteration']

        if self.iteration > self.patience:
            if self.loss is None or value < self.loss*self.imporovement:
                self.loss = value
                self.save_model(test_res)
                self.patience = self.iteration + self.interval
            else:
                return True
        return False
