import numpy
import six
import chainer
from chainer import cuda

from masalachai import Logger
from masalachai import DataFeeder
from masalachai import Trainer

class VirtualAdversarialTrainer(Trainer):
    def __init__(self, optimizer, labeled_data, unlabeled_data, test_data, gpu, eps=4.0, xi=0.1, lam=1., pitr=1,
                 logging=True, logfile=None, logcheryl=None, loguser=None):
        super(VirtualAdversarialTrainer, self).__init__(optimizer, labeled_data, test_data, gpu, 
                                                        logging=logging, logfile=logfile, logcheryl=logcheryl, loguser=loguser)
        self.unlabeled_data = unlabeled_data
        self.eps = eps
        self.xi = xi
        self.lam = lam
        self.pitr = pitr

    def unsupervised_update(self, batchsize):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.unlabeled_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx = chainer.Variable(xp.asarray(data['data']))

        # forward and update
        self.optimizer.update(self.lds, vx)
        return self.lds_loss.data

    def lds(self, x):
        def kl_divergence(p, q, eps=10e-8):
            return chainer.functions.sum(p * (chainer.functions.log(p+eps) - chainer.functions.log(q+eps))) / p.data.shape[0]

        xp = cuda.cupy.get_array_module(x.data)

        p = chainer.functions.softmax(self.optimizer.target.predictor(x))
        p.unchain_backward()

        # init d as a random unit vector
        d = xp.random.normal(size=x.data.shape).astype(xp.float32)
        d = d / xp.sqrt(xp.sum(d*d, axis=1)).reshape(d.shape[0], 1)

        # approximate r_vadv by power iteration method
        r = xp.zeros(x.data.shape, dtype=xp.float32)
        vr = chainer.Variable(r, volatile='off')
        for ip in six.moves.range(self.pitr):
            vd = chainer.Variable(d)
            q = chainer.functions.softmax(self.optimizer.target.predictor(x+vr+self.xi*d))
            k = kl_divergence(p, q)
            k.backward()
            d = vr.grad / xp.sqrt(xp.sum(vr.grad*vr.grad, axis=1)).reshape(d.shape[0], 1)
        self.r_vadv = chainer.Variable(self.eps * d)
        q_vadv = chainer.functions.softmax(self.optimizer.target.predictor(x+self.r_vadv))
        self.lds_loss = self.lam * kl_divergence(p, q_vadv)
        return self.lds_loss

    def train(self, nitr, lbatchsize, ubatchsize, log_interval=100, test_interval=100, test_batchsize=100, test_nitr=1):
        # training
        self.train_batch = self.train_data.batch(lbatchsize, shuffle=True)
        self.unlabeled_batch = self.unlabeled_data.batch(ubatchsize, shuffle=True)
        supervised_loss = 0.
        unsupervised_loss = 0.
        train_acc = 0.
        for i in six.moves.range(1, nitr+1):
            supervised_loss += float(self.supervised_update(lbatchsize))
            self.optimizer.target.loss.unchain_backward()
            train_acc += float(self.optimizer.target.accuracy.data)
            unsupervised_loss += float(self.unsupervised_update(ubatchsize))
            self.lds_loss.unchain_backward()
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0 and self.logging:
                self.logger.loss_acc_log(i, (supervised_loss+unsupervised_loss)/log_interval, train_acc/log_interval)
                supervised_loss = 0.
                unsupervised_loss = 0.
                train_acc = 0.

            # test
            if i % test_interval == 0 and self.logging:
                self.test(test_nitr, test_batchsize)

        # logging
        if self.logging:
            self.logger.loss_log(nitr, (supervised_loss+unsupervised_loss)/((nitr%log_interval)+1))
        # test
        if test_interval > 0 and self.logging:
            self.test(test_nitr, test_batchsize)
