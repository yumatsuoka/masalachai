import chainer

from masalachai import Logger
from masalachai import Trainer
from masalachai.datafeeders import SiameseFeeder

class SiasemseTrainer(Trainer):
    def __init__(self, optimizer, train_data, test_data, gpu,
                 logging=True, logfile=None, logcheryl=None, loguser=None):
        super(SiasemseTrainer, self).__init__(optimizer, trai_data, test_data, gpu,
                                              logging=logging, logfile=logfile, logcheryl=logcheryl, loguser=loguer)
        self.train_data = SiameseFeeder(data_dict=train_data)

    def supervised_update(self, batchsize):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx_0, vx_1 = chainer.Variable(xp.asarray(data['data']))
        vt = chainer.Variable(xp.asarray(data['target']))

        # forward and update
        # TODO: change
        #self.optimizer.update(self.optimizer.target, vx, vt)
        return self.optimizer.target.loss.data

    def train(self, nitr, batchsize, log_interval=100, test_interval=100, test_batchsize=100, test_nitr):
        pass
