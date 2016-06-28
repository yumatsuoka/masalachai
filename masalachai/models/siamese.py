# -*- coding: utf-8 -*-

from masalachai import Model
from chainer.functions import contrastive

## Siamese Network Wrapper
class SiameseModel(Model):
    def __init__(self, predictor, lossfun=contrastive):
        super(SiameseModel, self).__init__(predictor, lossfun)
        self.discriminator = masalachai.models.SupervisedModel(decoder)

    def __call__(self, x, t, train=True):
        self.y = None
        self.loss = None
        x0, x1, = x
        t0, = t
        self.y0 = self.predictor(x0, train=train)
        self.y1 = self.predictor(x1, train=train)
        self.loss = self.lossfun(self.y0, self.y1, t0)
        return self.loss

    def predict(self, x):
        x0, = x
        self.y = self.predictor(x0, train=False)
        return self.y

