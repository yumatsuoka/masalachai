import numpy
import six
import chainer
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder

class Trainer(object):
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

    def predict(self, batchsize, train=False):
        raise NotImplementedError

    def train(self, nitr, batchsize):
        raise NotImplementedError

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
