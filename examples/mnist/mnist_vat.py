#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
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
from masalachai.optimizer_schedulers import DecayOptimizerScheduler
from mlp import Mlp

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--epoch', '-e', type=int, default=10, help='training epoch (default: 10)')
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

# making labeled data
s_each = int(args.slabeled / 10)
sample_breakdown = np.array([s_each for i in six.moves.range(10)])
labeled_data_samples = np.zeros((sample_breakdown.sum(), dim), dtype=np.float32)
labeled_data_labels = np.zeros(sample_breakdown.sum(), dtype=np.int32)
sp = 0
for t in six.moves.range(s_each):
    idx = np.where(dataset['train']['target']==t)[0][:sample_breakdown[t]]
    ep = sp + sample_breakdown[t]
    labeled_data_samples[sp:ep] =  unlabeled_data_dict['data'][idx]
    labeled_data_labels[sp:ep] = dataset['train']['target'][idx]
    sp += sample_breakdown[t]
labeled_data_dict = {'data':labeled_data_samples,
                     'target':labeled_data_labels}


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


trainer = trainers.VirtualAdversarialTrainer(optimizer, logger, (labeled_data,unlabeled_data), test_data, args.gpu, eps=0.4, xi=0.001, lam=1.0)
trainer.add_optimizer_scheduler(adam_alpha_scheduler)
trainer.train(int(args.epoch*args.slabeled/args.lbatch), 
              log_interval=1,
              test_interval=100, 
              test_nitr=10)
