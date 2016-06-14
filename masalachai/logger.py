# -*- coding: utf-8 -*-

import logging
from cheryl import CherylAPI

class Logger(object):
    formatter = logging.Formatter(fmt='%(asctime)s %(name)8s %(levelname)8s: %(message)s',datefmt='%Y/%m/%d %p %I:%M:%S,',)
    handler = logging.StreamHandler()

    def __init__(self, name, level=logging.INFO, tofile=None, tocheryl=None, touser=None):
        # stream handler setting
        self.handler.setLevel(level)
        self.handler.setFormatter(self.formatter)

        # logger setting
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(self.handler)

        # file handler setting
        if tofile is not None:
            self.file_handler = logging.FileHandler(tofile)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

        # Cheryl API setting
        self.cheryl = (tocheryl is not None) and (touser is not None)
        if self.cheryl:
            self.bot = CherylAPI(configfile=tocheryl)
            self.user = touser


    def __call__(self, msg):
        self.logger.info(msg)
        if self.cheryl:
            self.bot.post_direct_message_by(self.user, '['+os.uname()[1]+'] '+msg)

    def log(self, itr, loss, acc=None, train=True):
        if train and acc is not None:
            log_str = '{0:d}, loss={1:.5f}, accuracy={2:.5f}'.format(itr, loss, acc)
        elif train and acc is None:
            log_str = '{0:d}, loss={1:.5f}'.format(itr, loss)
        elif not train and acc is not None:
            log_str = '[TEST], loss={0:.5f}, accuracy={1:.5f}'.format(loss, acc)
        elif not train and acc is None:
            log_str = '[TEST], loss={0:.5f}'.format(loss)
        self.__call__(log_str)


    def loss_acc_log(self, itr, loss, acc):
        log_str = '{0:d}, loss={1:.5f}, accuracy={2:.5f}'.format(itr, loss, acc)
        self.__call__(log_str)

    def loss_log(self, itr, loss):
        log_str = '{0:d}, loss={1:.5f}'.format(itr, loss)
        self.__call__(log_str)

    def test_log(self,loss, acc):
        log_str = '[TEST], loss={0:.5f}, accuracy={1:.5f}'.format(loss, acc)
        self.__call__(log_str)

