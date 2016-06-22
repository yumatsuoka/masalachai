import numpy
import six
import chainer
import queue
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder

class Trainer(object):
    queue_size = 3

    def __init__(self, optimizer, train_data, logger, gpu=-1):
        self.optimizer = optimizer
        self.train_data = train_data
        self.logger = logger
        self.gpu = gpu

        # 並列処理用のメモリを各モジュールに登録
        self._data_queue = queue.Queue(self.queue_size)
        self._log_queue = queue.Queue()
        self.train_data.setQueue(self._data_queue)
        self.logger.setQueue(self._log_queue)

        _optimizer_param_schedulers = []


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

    def update(self, batchsize):
        # パラメータ更新を実行
        raise NotImplementedError

    def train(self, nitr, batchsize, 
              log_interval=1, 
              test_interval=1, test_nitr=1, test_batchsize=1):
        # 学習を実行

        # setting stop event for data feeder
        stop_feeding = threading.Event()
        self.train_data.setEvent(stop_feeding)

        # start threads
        self.train_data.start()
        self.logger.start()

        for i in six.moves.range(1, nitr+1):
            # update
            res = update(batchsize)
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0:
                res['iteration'] = i
                _log_queue.put(self.logger.train_log_mode)
                _log_queue.put(res)
            # test
            if i % test_interval == 0:
                res = test(test_nitr, test_batchsize)
                _log_queue.put(self.logger.test_log_mode)
                _log_queue.put(res)

        stop_feeding.set()
        # clearn data queue
        while not self._data_queue.empty():
            self._data_queue.get()

        _log_queue.put('END')
        self.train_data.join()
        self.logger.join()


    def test(self, nitr, batchsize):
        # テストを実行
        raise NotImplementedError

