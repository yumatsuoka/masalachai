# -*- coding: utf-8 -*-

from chainer import link
from chainer.functions import contrastive

## Siamese Network Wrapper
class SiameseNet(link.Chain):
    def __init__(self, predictor, lossfun=contrastive):
        super(SiameseNet, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        #self.y = None
        self.loss = None

    def __call__(self, x0, x1, t):
        #self.y = None
        self.loss = None
        self.y0 = self.predictor(x0)
        self.y1 = self.predictor(x1)
        self.loss = self.lossfun(self.y0, self.y1, t)
        return self.loss
