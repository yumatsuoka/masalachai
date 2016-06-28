#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

# import dataset script
import mnist

# import model network
from masalachai import DataFeeder
from masalachai import Logger
from masalachai import trainers
from masalachai import models
from autoencoder import Perceptrons

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

# Logger setup
logger = Logger('MNIST AE',
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
N_test = len(dataset['test']['target'])
train_data_dict = {'data':dataset['train']['data'].reshape(N_train, dim).astype(np.float32)}
test_data_dict = {'data':dataset['test']['data'].reshape(N_test, dim).astype(np.float32)}
train_data = DataFeeder(train_data_dict)
test_data = DataFeeder(test_data_dict)

train_data.hook_preprocess(mnist_preprocess)
test_data.hook_preprocess(mnist_preprocess)


# Model Setup
h_units = 1200
model = models.AutoencoderModel(
            Perceptrons(train_data['data'][0].size, h_units, activation=F.relu),
            Perceptrons(h_units, train_data['data'][0].size)
        )
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Opimizer Setup
optimizer = optimizers.Adam()
optimizer.setup(model)

trainer = trainers.AutoencoderTrainer(optimizer, logger, (train_data,), test_data, args.gpu)
trainer.train(int(args.epoch*N_train/args.batch), 
              (args.batch,),
              test_batchsize=args.valbatch,
              test_interval=N_train/args.batch)

