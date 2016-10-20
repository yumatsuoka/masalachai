import numpy
import six
import chainer
from chainer import cuda, function

from masalachai.trainers.supervised_trainer import SupervisedTrainer

"""
class Sign(function.Function):
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype == numpy.float32,)

    def forward(self, x):
        y = x[0].copy()
        xp = cupy.get_array_module(y)
        y = xp.sign(y)
        return y,

    def backward(self, x, gx):
        gx = gy[0].copy()
        return gx,

def sign(x):
    return Sign()(x)
"""


class AdversarialTrainer(SupervisedTrainer):

    def __init__(self, optimizer, loggers,
                 train_data_feeders, test_data_feeder, gpu=-1,
                 eps=4.0, alpha=0.5):
        super(AdversarialTrainer, self).__init__(
            optimizer, loggers, train_data_feeders,
            test_data_feeder=test_data_feeder, gpu=gpu)
        self.eps = eps
        self.alpha = alpha

    def update(self):
        return self.supervised_update()

    def supervised_update(self):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        self.accuracy = None

        # read data
        data = self.train_data_queues[0].get()
        vx = tuple([chainer.Variable(xp.asarray(data[k]))
                    for k in data.keys() if 'data' in k])
        vt = tuple([chainer.Variable(xp.asarray(data[k]))
                    for k in data.keys() if 'target' in k])

        self.optimizer.update(self.adversarial_loss, vx, vt)

        # get result
        res = {'loss': float(self.loss.data),
               'adversarial_loss': float(self.adv_loss.data)}
        if self.accuracy is not None:
            res['accuracy'] = self.accuracy
        return res

    def adversarial_loss(self, vx, vt):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        self.loss = self.optimizer.target(vx, vt)
        if self.optimizer.target.accuracy is not None:
            self.accuracy = float(self.optimizer.target.accuracy.data)
        self.loss.backward()

        adv_vx = tuple([chainer.Variable(
            self.eps * xp.sign(x.grad) + x.data) for x in vx])
        self.adv_loss = self.optimizer.target(adv_vx, vt)

        return self.alpha * self.loss + (1 - self.alpha) * self.adv_loss
