# -*- coding: utf-8 -*-

from chainer import link
from chainer.functions import mean_squared_error

## Autoencoder Wrapper
class AutoencodeModel(link.Chain):
    def __init__(self, predictor, lossfun=mean_squared_error):
        super(AutoencodeModel, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.h = None
        self.y = None
        self.loss = None

    def __call__(self, x):
        self.y = None
        self.loss = None
        x0, = x
        self.y = self.predictor(x0)
        self.loss = self.lossfun(x0, self.y)
        return self.loss

    def predict(self, x):
        x0, = x
        self.h = self.predictor.encode(x0)
        return self.h

    def reconstruct(self, x):
        x0, = x
        self.h = self.predictor.decode(x0)
        return self.h

