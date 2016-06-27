import numpy
import six
import chainer
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder
from masalachai import Trainer

class SupervisedTrainer(Trainer):
    def __init__(self, optimizer, logger, train_data, test_data, gpu=-1):
        super(SupervisedTrainer, self).__init__(optimizer, logger, train_data, test_data=test_data, gpu=gpu)


    def supervised_update(self, batchsize):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_data_queue.get()
        vx = tuple( [ chainer.Variable( xp.asarray(data[k]) ) for k in data.keys() if 'data' in k ] )
        vt = tuple( [ chainer.Variable( xp.asarray(data[k]) ) for k in data.keys() if 'target' in k ] )

        # forward and update
        self.optimizer.update(self.optimizer.target, vx, vt)

        # get result
        res = {'loss': float(self.optimizer.target.loss.data),
               'accuracy': float(self.optimizer.target.accuracy.data)}
        return res

    def update(self, batchsize):
        return self.supervised_update(batchsize)


    def predict(self, batchsize, train=False):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.test_data_queue.get()
        vx = tuple( [ chainer.Variable( xp.asarray(data[k]), volatile='on' ) for k in data.keys() if 'data' in k ] )
        vt = tuple( [ chainer.Variable( xp.asarray(data[k]), volatile='on' ) for k in data.keys() if 'target' in k ] )

        # forward
        self.optimizer.target(vx, vt, train=train)

        # get result
        res = {'loss': float(self.optimizer.target.loss.data),
               'accuracy': float(self.optimizer.target.accuracy.data)}
        return res

