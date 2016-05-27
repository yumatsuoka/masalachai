#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers

# import dataset script
import cifar10

# import model network
from masalachai import Trainer
from allconvnet import AllConvNet
from allconvnet_bn import AllConvNetBN

# argparse
parser = argparse.ArgumentParser(description='All Convolutional Network Example on CIFAR-10')
parser.add_argument('--epoch', '-e', type=int, default=100, help='training epoch (default: 100)')
parser.add_argument('--batch', '-b', type=int, default=100, help='training batchsize (default: 100)')
parser.add_argument('--valbatch', '-v', type=int, default=100, help='validation batchsize (default: 100)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def cifar_preprocess(data):
    data['data'] /= 255.
    return data

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = cifar10.load()

train_data = dataset['train']
train_data['data'] = train_data['data'].astype(np.float32)
train_data['target'] = train_data['target'].astype(np.int32)

test_data = dataset['test']
test_data['data'] = test_data['data'].astype(np.float32)
test_data['target'] = test_data['target'].astype(np.int32)


# Model Setup
model = L.Classifier(AllConvNet())
#model = L.Classifier(AllConvNetBN())
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Opimizer Setup
optimizer = optimizers.Adam()
optimizer.setup(model)

trainer = Trainer(optimizer, train_data, test_data, args.gpu)
trainer.hook(cifar_preprocess)
trainer.train(args.epoch*train_data['size']/args.batch, 
              args.batch, 
              test_interval=train_data['size']/args.batch, 
              test_nitr=test_data['size']/args.valbatch,
              test_batchsize=args.valbatch)
