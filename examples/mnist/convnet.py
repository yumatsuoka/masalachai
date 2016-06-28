# -*- coding: utf-8 -*-

# Convolutional Network for MNIST

import chainer
import chainer.functions as F
import chainer.links as L

class ConvNet(chainer.Chain):

    def __init__(self, in_channels=1, output=10):
        super(ConvNet, self).__init__(
                conv1 = L.Convolution2D(in_channels, 16, 3, pad=1),
                conv2 = L.Convolution2D(16, 16, 3, pad=1),
                conv3 = L.Convolution2D(16, 16, 3, pad=1),
                fc1 = L.Linear(144, output),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d( F.relu(self.conv1(x)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv2(h)), 3, stride=2 )
        h = F.max_pooling_2d( F.relu(self.conv3(h)), 3, stride=2 )
        h = self.fc1(h)
        return h
