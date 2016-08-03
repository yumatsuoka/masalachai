import numpy
import chainer
from chainer import cuda

from masalachai.trainer import Trainer

class AutoencoderTrainer(Trainer):
    def __init__(self, optimizer, logger, train_data_feeders, test_data_feeder=None, gpu=-1):
        super(AutoencoderTrainer, self).__init__(optimizer, logger, train_data_feeders, test_data_feeder=test_data_feeder, gpu=gpu)

    def update(self):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_data_queues[0].get()
        vx = tuple( [ chainer.Variable( xp.asarray(data[k]) ) for k in data.keys() if 'data' in k ] )

        # forward and update
        self.optimizer.update(self.optimizer.target, vx)

        # get result
        res = {'loss': float(self.optimizer.target.loss.data)}
        return res

    def predict(self, train=False):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.test_data_queue.get()
        vx = tuple( [ chainer.Variable( xp.asarray(data[k]), volatile='on' ) for k in data.keys() if 'data' in k ] )

        # forward
        self.optimizer.target(vx, train=train)

        # get result
        res = {'loss': float(self.optimizer.target.loss.data)}
        return res

