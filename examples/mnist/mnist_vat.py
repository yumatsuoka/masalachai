#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import argparse
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers
from sklearn import cross_validation

# import dataset script
import mnist

# import model network
from masalachai import DataFeeder
from masalachai import Logger
from masalachai import models
from masalachai.optimizer_schedulers import DecayOptimizerScheduler
from mlp import Mlp

# for test
from masalachai.trainers import VirtualAdversarialTrainer
#from vat import VirtualAdversarialTrainer

test_norm = 'kl_d'
#test_norm = 'euclidean_d'

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--nitr', '-n', type=int, default=10000, help='number of times of weight update (default: 10000)')
parser.add_argument('--lbatch', '-l', type=int, default=100, help='labeled training batchsize (default: 100)')
parser.add_argument('--ubatch', '-u', type=int, default=250, help='unlabeled training batchsize (default: 250)')
parser.add_argument('--valbatch', '-v', type=int, default=1000, help='validation batchsize (default: 1000)')
parser.add_argument('--slabeled', '-s', type=int, default=100, help='size of labeled training samples (default: 100)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preprocess(data):
    data['data'] /= 255.
    return data

# Logger setup
def vat_train_log(res):
    log_str = '{0:d}, loss={1:.5f}, lds={2:.5f}, accuracy={3:.5f}'.format(res['iteration'], res['loss'], res['lds'], res['accuracy'])
    return log_str

logger = Logger('MNIST MLP', train_log_mode='TRAIN_VAT')
logger.mode['TRAIN_VAT'] = vat_train_log

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
N_train = len(dataset['train']['target'])
N_test = len(dataset['test']['target'])
test_data_dict = {'data':dataset['test']['data'].reshape(N_test, dim).astype(np.float32),
                  'target':dataset['test']['target'].astype(np.int32)}
unlabeled_data_dict = {'data':dataset['train']['data'].reshape(N_train, dim).astype(np.float32)}
# making labeld data
lplo = cross_validation.LeavePLabelOut(labels=six.moves.range(N_train), p=args.slabeled)
fold = 1
for i in six.moves.range(fold):
    train_idx, test_idx = next(iter(lplo))
labeled_data_dict = {'data':unlabeled_data_dict['data'][test_idx].astype(np.float32),
                     'target':dataset['train']['target'][test_idx].astype(np.int32)}

labeled_data = DataFeeder(labeled_data_dict, batchsize=args.lbatch)
unlabeled_data = DataFeeder(unlabeled_data_dict, batchsize=args.ubatch)
test_data = DataFeeder(test_data_dict, batchsize=args.valbatch)

labeled_data.hook_preprocess(mnist_preprocess)
unlabeled_data.hook_preprocess(mnist_preprocess)
test_data.hook_preprocess(mnist_preprocess)


# Model Setup
h_units = 1200
model = models.ClassifierModel(Mlp(labeled_data['data'][0].size, h_units, h_units, np.max(labeled_data['target'])+1))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Opimizer Setup
optimizer = optimizers.Adam()
optimizer.setup(model)

alpha_decay_interval = 500
alpha_decay_rate = 0.9
adam_alpha_scheduler = DecayOptimizerScheduler(optimizer, 'alpha', alpha_decay_interval, alpha_decay_rate)


trainer = VirtualAdversarialTrainer(optimizer, logger, (labeled_data,unlabeled_data), test_data, args.gpu, eps=0.4, xi=0.001, lam=1.0, norm=test_norm)
trainer.add_optimizer_scheduler(adam_alpha_scheduler)
trainer.train(args.nitr, 
              log_interval=200,
              test_interval=2000, 
              test_nitr=1000)
