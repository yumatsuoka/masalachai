
from masalachai import Trainer

class AutoencodeTrainer(Trainer):
    def __init__(self, optimizer, data_feeder, gpu, logging=True, logfile=None, logcheryl=None, loguser=None):
        self.optimizer = optimizer
        self.train_data = data_feeder
        self.gpu = gpu
        self.logging = logging
        if self.logging:
            self.logger = Logger(__name__, tofile=logfile, tocheryl=logcheryl, touser=loguser)

    def supervised_update(self, batchsize):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.train_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx = tuple([chainer.Variable(xp.asarray(d)) for d in data['data']])

        # forward and update
        self.optimizer.update(self.optimizer.target, vx)
        return self.optimizer.target.loss.data

    def predict(self, batchsize, train=False):
        # array backend
        xp = cuda.cupy if self.gpu >= 0 else numpy

        # read data
        data = self.test_batch.next()
        for func in self._preprocess_hooks:
            data = func(data)
        vx = tuple([chainer.Variable(xp.asarray(d), volatile='on') for d in data['data']])

        # forward
        self.optimizer.target.predictor.train = train
        ret = self.optimizer.target.encode(vx)
        self.optimizer.target.predictor.train = not train
        return ret

    def train(self, nitr, batchsize, log_interval=100, test_interval=1000, test_batchsize=100, test_nitr=1):
        # training
        self.train_batch = self.train_data.batch(batchsize, shuffle=True)
        supervised_loss = 0.
        for i in six.moves.range(1, nitr+1):
            supervised_loss += float(self.supervised_update(batchsize))
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0 and self.logging:
                self.logger.loss_log(i, supervised_loss/log_interval)
                supervised_loss = 0.

        # logging
        if self.logging:
            self.logger.loss_log(nitr, supervised_loss / ((nitr%log_interval)+1))
