import numpy
import six
import chainer
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder
from masalachai import Trainer

class SupervisedTrainer(Trainer):
    _preprocess_hooks = []
    _optimizer_param_schedulers = []

    def __init__(self, optimizer, train_data, test_data, gpu, logging=True, logfile=None, logcheryl=None, loguser=None):
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.gpu = gpu
        self.logging = logging
        if self.logging:
            self.logger = Logger(__name__, tofile=logfile, tocheryl=logcheryl, touser=loguser)

    def hook(self, func):
        self._preprocess_hooks.append(func)

    def add_optimizer_scheduler(self, s):
        self._optimizer_param_schedulers.append(s)

    def optimizer_param_process(self, t):
        for s in self._optimizer_param_schedulers:
            self.optimizer.__dict__[s.param_name] = s.next(t)

    def supervised_update(self, batchsize):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx = tuple([chainer.Variable(xp.asarray(d)) for d in data['data']])
        vt = tuple([chainer.Variable(xp.asarray(t)) for t in data['target']])

        # forward and update
        self.optimizer.update(self.optimizer.target, vx, vt)
        return self.optimizer.target.loss.data


    def predict(self, batchsize, train=False):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.test_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx = tuple([chainer.Variable(xp.asarray(d), volatile='on') for d in data['data']])
        vt = tuple([chainer.Variable(xp.asarray(t), volatile='on') for t in data['target']])

        # forward
        self.optimizer.target.predictor.train = train
        self.optimizer.target(vx, vt)
        self.optimizer.target.predictor.train = not train


    def train(self, nitr, batchsize, log_interval=100, test_interval=1000, test_batchsize=100, test_nitr=1):
        # training
        self.train_batch = self.train_data.batch(batchsize, shuffle=True)
        supervised_loss = 0.
        train_acc = 0.
        for i in six.moves.range(1, nitr+1):
            supervised_loss += float(self.supervised_update(batchsize))
            if train_acc:
                train_acc += float(self.optimizer.target.accuracy.data) if self.optimizer.target.accuracy is not None else 0.
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0 and self.logging:
                self.logger.loss_acc_log(i, supervised_loss/log_interval, train_acc/log_interval)
                supervised_loss = 0.
                train_acc = 0.

            # test
            if i % test_interval == 0 and self.logging:
                self.test(test_nitr, test_batchsize)

        # logging
        if self.logging:
            self.logger.loss_log(nitr, supervised_loss / ((nitr%log_interval)+1))
        # test
        if test_interval > 0 and self.logging:
            self.test(test_nitr, test_batchsize)


    def test(self, nitr, batchsize):
        # testing
        acc = 0.
        loss = 0.
        self.test_batch = self.test_data.batch(batchsize, shuffle=False)
        for i in six.moves.range(nitr):
            self.predict(batchsize)
            acc += float(self.optimizer.target.accuracy.data) if self.optimizer.target.accuracy is not None else 0.
            loss += self.optimizer.target.loss.data
        # logging
        if self.logging:
            self.logger.test_log(float(loss/nitr), float(acc/nitr))
        return loss/nitr, acc/nitr
