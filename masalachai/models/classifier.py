# -*- coding: utf-8 -*-

from masalachai.model import Model
from chainer.functions import softmax_cross_entropy
from chainer.functions import accuracy

## classification network wapper
class ClassifierModel(Model):
    def __init__(self, predictor, lossfun=softmax_cross_entropy, accuracyfun=accuracy):
        super(ClassifierModel, self).__init__(predictor, lossfun)

    def __call__(self, x, t, train=True, compute_accuracy=True):
        self.loss = None
        self.accuracy = None
        x0, = x
        t0, = t
        self.y = self.predictor(x0, train=train)
        self.loss = self.lossfun(self.y, t0)
        if compute_accuracy:
            self.accuracy = accuracy(self.y, t0)
        return self.loss

    def predict(self, x):
        x0, = x
        self.y = self.predictor(x0, train=False)
        return self.y

