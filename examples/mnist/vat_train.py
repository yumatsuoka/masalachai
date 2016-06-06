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
from masalachai.optimizer_schedulers import DecayOptimizerScheduler
from masalachai import Trainer
from masalachai.trainers import VirtualAdversarialTrainer
from mlp import Mlp

# argparse
parser = argparse.ArgumentParser(description='Virtual Adversarial Trainer Example')
parser.add_argument('--itr', '-i', type=int, default=50000, help='training iteration (default: 50000)')
parser.add_argument('--nl', '-n', type=int, default=100, help='training labeled datasize (default: 100)')
parser.add_argument('--lbatch', '-b', type=int, default=100, help='training labeled batchsize (default: 100)')
parser.add_argument('--ubatch', '-u', type=int, default=250, help='training unlabeled batchsize (default: 250)')
parser.add_argument('--valbatch', '-v', type=int, default=100, help='validation batchsize (default: 100)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preproces(data):
    data['data'] /= 255.
    return data


# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
categories = np.max(dataset['train']['target'])+1
N_train = len(dataset['train']['target'])
N_test = len(dataset['test']['target'])

unlabeled_data = {'data':dataset['train']['data'].reshape(N_train, dim).astype(np.float32)}
test_data = {'data':dataset['test']['data'].reshape(N_test, dim).astype(np.float32),
             'target':dataset['test']['target'].astype(np.int32)}

# making labeled data
#sample_breakdown = np.array([2, 12, 12, 12, 12, 12, 12, 2, 12, 12])
sample_breakdown = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
labeled_data_samples = np.zeros((sample_breakdown.sum(), dim), dtype=np.float32)
labeled_data_labels = np.zeros(sample_breakdown.sum(), dtype=np.int32)
sp = 0
for t in xrange(categories):
    idx = np.where(dataset['train']['target']==t)[0][:sample_breakdown[t]]
    ep = sp + sample_breakdown[t]
    labeled_data_samples[sp:ep] =  unlabeled_data['data'][idx]
    labeled_data_labels[sp:ep] = dataset['train']['target'][idx]
    sp += sample_breakdown[t]

labeled_data = {'data':labeled_data_samples,
                'target':labeled_data_labels}

# Model Setup
h_units = 1200
model = L.Classifier(Mlp(dim, h_units, h_units, categories))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Opimizer Setup
optimizer = optimizers.Adam()
optimizer.setup(model)

alpha_decay_interval = 500
alpha_decay_rate = 0.9
adam_alpha_scheduler = DecayOptimizerScheduler(optimizer, 'alpha', alpha_decay_interval, alpha_decay_rate)


eps = 0.4
xi = 0.001
lam = 1.0

#logfile = 'vat_train_biased_plain_log.csv'
#logfile = 'vat_train_log.csv'
trainer = VirtualAdversarialTrainer(optimizer, labeled_data, unlabeled_data, test_data, args.gpu, eps=eps, xi=xi, lam=lam, logfile=None)
#trainer = Trainer(optimizer, labeled_data, test_data, args.gpu)
trainer.hook(mnist_preproces)
trainer.add_optimizer_scheduler(adam_alpha_scheduler)
trainer.train(args.itr, args.lbatch, args.ubatch, 
              test_interval=N_train/args.ubatch, 
              test_nitr=N_test/args.valbatch,
              test_batchsize=args.valbatch)
'''
trainer.train(args.itr, args.lbatch,
              test_interval=1000, 
              test_nitr=N_test/args.valbatch,
              test_batchsize=args.valbatch)
'''

# visualize virtual adversarial examples
import scipy.misc
x = chainer.Variable(xp.asarray(unlabeled_data['data'][4000:4005]).astype(xp.float32))
trainer.lds(x)
print xp.max(x.data / 255.)
x_vadv = (x.data/255. + trainer.r_vadv.data/eps).reshape(x.data.shape[0], 28, 28)
r_vadv = trainer.r_vadv.data.reshape(x.data.shape[0], 28, 28)
x_vadv = cuda.to_cpu(x_vadv)
r_vadv = cuda.to_cpu(r_vadv)
for itr, im in enumerate(x_vadv):
    scipy.misc.imsave('xvadv_'+str(itr)+'.png', im)
for itr, im in enumerate(r_vadv):
    scipy.misc.imsave('rvadv_'+str(itr)+'.png', im)

trainer.train(3000, args.lbatch, args.ubatch, 
              test_interval=N_train/args.ubatch, 
              test_nitr=N_test/args.valbatch,
              test_batchsize=args.valbatch)

trainer.lds(x)
x_vadv = (x.data/255. + trainer.r_vadv.data/eps).reshape(x.data.shape[0], 28, 28)
r_vadv = trainer.r_vadv.data.reshape(x.data.shape[0], 28, 28)
x_vadv = cuda.to_cpu(x_vadv)
r_vadv = cuda.to_cpu(r_vadv)
for itr, im in enumerate(x_vadv):
    scipy.misc.imsave('after_xvadv_'+str(itr)+'.png', im)
for itr, im in enumerate(r_vadv):
    scipy.misc.imsave('after_rvadv_'+str(itr)+'.png', im)
