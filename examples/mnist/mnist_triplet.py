#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers
from sklearn import cross_validation

# import dataset script
import mnist

# import model network
from masalachai import models
from masalachai import datafeeders
from masalachai import Logger
from masalachai import trainers
from convnet import ConvNet

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--nitr', '-n', type=int, default=10000, help='number of times of weight update (default: 10000)')
parser.add_argument('--batch', '-b', type=int, default=100, help='training batchsize (default: 100)')
parser.add_argument('--valbatch', '-v', type=int, default=100, help='validation batchsize (default: 1000)')
parser.add_argument('--slabeled', '-s', type=int, default=1000, help='number of labeled data  (default: 1000)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preprocess(data):
    data['data0'] /= 255.
    data['data1'] /= 255.
    data['data2'] /= 255.

    data['data0'] = np.expand_dims(data['data0'], 0)
    data['data1'] = np.expand_dims(data['data1'], 0)
    data['data2'] = np.expand_dims(data['data2'], 0)
    return data

# Logger setup
logger = Logger('MNIST TRIPLET',
                train_log_mode='TRAIN_LOSS_ONLY',
                test_log_mode='TEST_LOSS_ONLY')

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
N_train = len(dataset['train']['target'])
N_l = args.slabeled
N_ul = N_train - args.slabeled
train_data_dict = {'data':dataset['train']['data'].astype(np.float32),
                 'target':dataset['train']['target'].astype(np.int32)}

# make labeled and unlabeled data in training data
lplo = cross_validation.LeavePLabelOut(labels=six.moves.range(N_train), p=args.slabeled)
fold = 1
for i in six.moves.range(fold):
    ul_idxes, l_idxes = next(iter(lplo))
labeled_data_dict = {'data':train_data_dict['data'][l_idxes],
                    'target':train_data_dict['target'][l_idxes]}
ulabeled_data_dict ={'data':train_data_dict['data'][ul_idxes],
                   'target':train_data_dict['target'][ul_idxes]}

train_data = datafeeders.TripletFeeder(labeled_data_dict, batchsize=args.batch)
test_data =  datafeeders.TripletFeeder(ulabeled_data_dict, batchsize=args.valbatch)

train_data.hook_preprocess(mnist_preprocess)
test_data.hook_preprocess(mnist_preprocess)


# Model Setup
outputs = 2
model = models.TripletModel(ConvNet(output=outputs))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Opimizer Setup
optimizer = optimizers.Adam(alpha=0.0001)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))


trainer = trainers.SupervisedTrainer(optimizer, logger, (train_data,), None, args.gpu)
trainer.train(args.nitr, 1, 100, 1)
