import numpy
import six
import chainer
import queue
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder

class Trainer(object):
    # モデルの学習を制御するクラス
    _optimizer_param_schedulers = []

    def __init__(self, optimizer, train_data, logger, gpu=-1):
        self.optimizer = optimizer
        self.train_data = train_data
        self.logger = logger
        self.gpu = gpu

        # 並列処理用のメモリを各モジュールに登録
        self._data_queue = queue.Queue()
        self._log_queue = queue.Queue()
        self.train_data.setQueue(self._data_queue)
        self.logger.setQueue(self._log_queue)


    def add_optimizer_scheduler(self, s):
        # Optimizer のパラメータスケジューラの登録
        self._optimizer_param_schedulers.append(s)

    def optimizer_param_process(self, t):
        # Optimizer のパラメータスケジューラの駆動
        for s in self._optimizer_param_schedulers:
            self.optimizer.__dict__[s.param_name] = s.next(t)


    def predict(self, batchsize, train=False):
        # 推論を実行
        raise NotImplementedError

    def train(self, nitr, batchsize):
        # 学習を実行
        self.logger.start()
        # something
        _log_queue.put('END')
        self.logger.join()
        raise NotImplementedError

    def test(self, nitr, batchsize):
        # テストを実行
        raise NotImplementedError

