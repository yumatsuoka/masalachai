# -*- coding: utf-8 -*-

from chainer import link
from chainer.functions import identity

## Base Model Wrapper
class Model(link.Chain):
    def __init__(self, predictor, lossfun=identity):
        super(Model, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, x, train=True):
        self.y = None
        self.loss = None
        x0, = x
        self.y = self.predictor(x0, train=train)
        self.loss = self.lossfun(self.y)
        return self.loss

    def predict(self, x):
        x0, = x
        self.y = self.predictor(x0, train=False)
        return self.y
