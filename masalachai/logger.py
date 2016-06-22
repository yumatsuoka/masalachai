# -*- coding: utf-8 -*-

import os
import logging
import threading

class Logger(threading.Thread):
    formatter = logging.Formatter(fmt='%(asctime)s %(name)8s %(levelname)8s: %(message)s',datefmt='%Y/%m/%d %p %I:%M:%S,',)

    def __init__(self, name, level=logging.INFO, logfile=None, train_log_mode='TRAIN', test_log_mode='TEST'):
        super(Logger, self).__init__()

        # stream handler setting
        self.handler = logging.StreamHandler()
        self.handler.setLevel(level)
        self.handler.setFormatter(self.formatter)

        # logger setting
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(self.handler)

        # file handler setting
        if logfile is not None:
            self.file_handler = logging.FileHandler(logfile)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
        
        self.mode = {'TRAIN': self.train_log, 'TRAIN_LOSS_ONLY': self.train_loss_log, 'TEST': self.test_log, 'TEST_LOSS_ONLY': self.test_loss_log, 'END': self.log_end}
        self.train_log_mode = train_log_mode
        self.test_log_mode = test_log_mode
        self.queue = None
        self.stop = None


    def __call__(self, msg):
        self.logger.info(msg)


    def setQueue(self, queue):
        self.queue = queue


    def run(self):
        # queue check
        assert self.queue is None, "Log Queue is None, use Logger.setQueue(queue) before calling me."

        self.stop = threading.Event()
        while not self.stop.is_set():
            res = self.queue.get()
            if res in self.mode:
                log_func = self.mode[res]
                continue
            log_func(res)


    def log_train(self, res):
        log_str = '{0:d}, loss={1:.5f}, accuracy={2:.5f}'.format(res['iteration'], res['loss'], res['accuracy'])
        self.__call__(log_str)

    def log_train_loss_only(self, res):
        log_str = '{0:d}, loss={1:.5f}'.format(res['iteration'], res['loss'])

        self.__call__(log_str)

    def log_test(self, res):
        log_str = '[TEST], loss={0:.5f}, accuracy={1:.5f}'.format(res['loss'], res['accuracy'])
        self.__call__(log_str)

    def log_test_loss_only(self, res):
        log_str = '[TEST], loss={0:.5f}'.format(res['loss'])
        self.__call__(log_str)

    def log_end(self, *args):
        self.stop.set()

