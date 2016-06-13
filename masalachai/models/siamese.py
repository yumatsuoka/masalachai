# -*- coding: utf-8 -*-

from chainer import link
from chainer.functions import contrastive

## Siamese Network Wrapper
class SiameseModel(link.Chain):
    def __init__(self, predictor, lossfun=contrastive):
        super(SiameseModel, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        #self.y = None
        self.loss = None

    def __call__(self, x, t):
        #self.y = None
        self.loss = None
        x0, x1 = x
        t0, = t
        self.y0 = self.predictor(x0)
        self.y1 = self.predictor(x1)
        self.loss = self.lossfun(self.y0, self.y1, t0)
        return self.loss
