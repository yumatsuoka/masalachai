#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers

# import dataset script
import mnist

# import model network
from masalachai import Trainer
from mlp import Mlp

# for logging
from logging import getLogger,Formatter,StreamHandler,INFO
logger = getLogger(__name__)
formatter = Formatter(fmt='%(asctime)s %(message)s',datefmt='%Y/%m/%d %p %I:%M:%S,',)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(formatter)
logger.setLevel(INFO)
logger.addHandler(handler)

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--epoch', '-e', type=int, default=10, help='training epoch (default: 10)')
parser.add_argument('--batch', '-b', type=int, default=300, help='training batchsize (default: 300)')
parser.add_argument('--valbatch', '-v', type=int, default=1000, help='validation batchsize (default: 1000)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preprocess(data):
    data['data'] /= 255.
    return data

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
N_train = len(dataset['train']['target'])
N_test = len(dataset['test']['target'])
train_data = {'data':dataset['train']['data'].reshape(N_train, dim).astype(np.float32),
              'target':dataset['train']['target'].astype(np.int32)}
test_data = {'data':dataset['test']['data'].reshape(N_test, dim).astype(np.float32),
             'target':dataset['test']['target'].astype(np.int32)}

# Model Setup
h_units = 1200
model = L.Classifier(Mlp(train_data['data'][0].size, h_units, h_units, np.max(train_data['target'])+1))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Opimizer Setup
optimizer = optimizers.Adam()
optimizer.setup(model)

trainer = Trainer(optimizer, train_data, test_data, args.gpu)
trainer.hook(mnist_preprocess)
trainer.train(args.epoch*N_train/args.batch, 
              args.batch, 
              test_interval=N_train/args.batch, 
              test_nitr=N_test/args.valbatch,
              test_batchsize=args.valbatch)
