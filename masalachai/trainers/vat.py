import numpy
import six
import chainer
from chainer import cuda

from masalachai import Trainer

class VirtualAdversarialTrainer(Trainer):
    def __init__(self, optimizer, labeled_data, unlabeled_data, test_data, gpu, eps=4.0, xi=0.1, pitr=1):
        super(VirtualAdversarialTrainer, self).__init__(optimizer, labeled_data, test_data, gpu)
        self.unlabeled_data = DataFeeder(data_dict=unlabeled_data)
        self.eps = eps
        self.xi = xi
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
            return F.sum(p * (F.log(p+eps) - F.log(q+eps))) / p.data.shape[0]

        xp = cuda.cupy.get_array_module(x.data)

        p = F.softmax(self.optimizer.target.predictor(x))
        p.unchain_backward()

        # init d as a random unit vector
        d = xp.random.normal(size=x.data.shape).astype(xp.float32)
        d = d / xp.sqrt(xp.sum(d*d, axis=1)).reshape(d.shape[0], 1)

        # approximate r_vadv by power iteration method
        r = xp.zeros(x.data.shape, dtype=xp.float32)
        vr = chainer.Variable(r, volatile='off')
        for ip in six.moves.range(self.pitr):
            vd = chainer.Variable(d)
            q = F.softmax(self.optimizer.target.predictor(x+vr+self.xi*d))
            k = kl_divergence(p, q)
            k.backward()
            d = vr.grad / xp.sqrt(xp.sum(vr.grad*vr.grad, axis=1)).reshape(d.shape[0], 1)
        r_vadv = chainer.Variable(self.eps * d)
        q_vadv = F.softmax(self.optimizer.target.predictor(x+r_vadv))
        self.lds_loss = kl_divergence(p, q_vadv)
        return self.lds_loss

    def train(self, nitr, lbatchsize, ubatchsize, log_interval=100, test_flag=True, test_batch=100, test_nitr=None):
        if test_flag and test_nitr is None:
            test_nitr = self.test_data.n / test_batch
        # training
        self.train_batch = self.train_data.batch(lbatchsize, shuffle=True)
        self.unlabeled_batch = self.unlabeled_data.batch(ubatchsize, shuffle=True)
        supervised_loss = 0.
        unsupervised_loss = 0.
        for i in six.moves.range(nitr):
            supervised_loss += self.supervised_update(lbatchsize)
            self.optimizer.target.loss.unchain_backward()
            unsupervised_loss += self.unsupervised_update(ubatchsize)
            self.lds_loss.unchain_backward()
            self.optimizer_param_process(i)
            if i % log_interval == 0:
                log = str(i) + ', ' + str(supervised_loss/log_interval) + ', ' + str(unsupervised_loss/log_interval)
                if test_flag:
                    test_acc, test_loss = self.test(test_nitr, test_batch)
                    log += ', ' + str(test_acc) + ', ' + str(test_loss)
                print log
                unsupervised_loss = 0.
                supervised_loss = 0.
