# -*- coding: utf-8 -*-

# All Convolutional Network with Batch Normalization for CIFAR-10

import chainer
import chainer.functions as F
import chainer.links as L

class AllConvNetBN(chainer.Chain):

    def __init__(self):
        super(AllConvNetBN, self).__init__(
                conv1 = L.Convolution2D(3, 96, 3),
                conv2 = L.Convolution2D(96, 96, 3, pad=1),
                bn1 = L.BatchNormalization(96),
                conv3 = L.Convolution2D(96, 96, 3, stride=2),
                conv4 = L.Convolution2D(96, 192, 3, pad=1),
                conv5 = L.Convolution2D(192, 192, 3, pad=1),
                bn2 = L.BatchNormalization(192),
                conv6 = L.Convolution2D(192, 192, 3, stride=2),
                conv7 = L.Convolution2D(192, 192, 3, pad=1),
                bn3 = L.BatchNormalization(192),
                conv8 = L.Convolution2D(192, 192, 1),
                conv9 = L.Convolution2D(192, 10, 1),
        )

    def __call__(self, x, t=None, train=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.bn1(self.conv2(h)))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.bn2(self.conv5(h)))
        h = F.relu(self.conv6(h))
        h = F.relu(self.bn3(self.conv7(h)))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 10))
        return h
