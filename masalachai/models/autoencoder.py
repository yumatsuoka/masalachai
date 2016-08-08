# -*- coding: utf-8 -*-

from masalachai.model import Model
from chainer.functions import mean_squared_error

# Autoencoder Wrapper


class AutoencoderModel(Model):

    def __init__(self, encoder, decoder, lossfun=mean_squared_error):
        super(AutoencoderModel, self).__init__(
            encoder, decoder=decoder, lossfun=lossfun)
        self.decoder = decoder
        self.z = None

    def __call__(self, x, train=True):
        self.y = None
        self.loss = None
        x0, = x
        self.y = self.predictor(x0, train=train)
        self.z = self.decoder(self.y, train=train)
        self.loss = self.lossfun(x0, self.z)
        return self.loss

    def decode(self, y, train=False):
        y0, = y
        self.z = self.decoder(y0, train=train)
        return self.z
