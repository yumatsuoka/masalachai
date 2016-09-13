import numpy
import six
import chainer
from chainer import cuda

from masalachai.trainers.supervised_trainer import SupervisedTrainer

def _axis_tuple(ary):
    return tuple([i for i in six.moves.range(ary.ndim)])

class VirtualAdversarialTrainer(SupervisedTrainer):
    def __init__(self, optimizer, logger, train_data_feeders, test_data_feeder, gpu=-1,
                 eps=4.0, xi=0.1, lam=1., pitr=1, norm='kl_d'):
        super(VirtualAdversarialTrainer, self).__init__(optimizer, logger, train_data_feeders, test_data_feeder=test_data_feeder, gpu=gpu)
        self.eps = eps
        self.xi = xi
        self.lam = lam
        self.pitr = pitr
        self.norm = norm

    def update(self):
        res_supervised = self.supervised_update()
        res_unsupervised = self.unsupervised_update()
        res_supervised.update(res_unsupervised)
        return res_supervised


    def unsupervised_update(self):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_data_queues[1].get()
        vx = tuple( [ chainer.Variable( xp.asarray(data[k]) ) for k in data.keys() if 'data' in k ] )

        # forward and update
        self.optimizer.update(self.lds, vx)

        res = {'lds': float(self.lds_loss.data)}
        return res


    def lds(self, x):
        def kl_divergence(p, q, eps=10e-8):
            return chainer.functions.sum(p * (chainer.functions.log(p+eps) - chainer.functions.log(q+eps))) / p.data.shape[0]

        def euclidean_d(p, q):
            return chainer.functions.sum(chainer.functions.sqrt((p-q) ** 2), axis=1)
        
        calc_d = (euclidean_d)if(self.norm == 'euclidean_d')else(kl_divergence)

        xp = cuda.cupy if self.gpu >= 0 else numpy

        x0, = x

        p = chainer.functions.softmax(self.optimizer.target.predictor(x0))
        p.unchain_backward()

        # init d as a random unit vector
        d = xp.random.normal(size=x0.data.shape).astype(xp.float32)
        #d = d / xp.sqrt(xp.sum(d*d, axis=d.shape[1:])).reshape(d.shape[0], 1)
        d = d / xp.sqrt(xp.sum(d*d, axis=_axis_tuple(d)[1:], keepdims=True))

        # approximate r_vadv by power iteration method
        r = xp.zeros(x0.data.shape, dtype=xp.float32)
        vr = chainer.Variable(r, volatile='off')
        for ip in six.moves.range(self.pitr):
            vd = chainer.Variable(d)
            q = chainer.functions.softmax(self.optimizer.target.predictor(x0+vr+self.xi*d))
            k = calc_d(p, q)
            k.backward()
            d = vr.grad / xp.sqrt(xp.sum(vr.grad*vr.grad, axis=_axis_tuple(d)[1:], keepdims=True))
        self.r_vadv = chainer.Variable(self.eps * d)
        q_vadv = chainer.functions.softmax(self.optimizer.target.predictor(x0+self.r_vadv))
        self.lds_loss = self.lam * calc_d(p, q_vadv)
        return self.lds_loss

