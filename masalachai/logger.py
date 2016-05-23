# -*- coding: utf-8 -*-

import logging

class Logger(object):
    formatter = logging.Formatter(fmt='%(asctime)s %(name)8s %(levelname)8s: %(message)s',datefmt='%Y/%m/%d %p %I:%M:%S,',)
    handler = logging.StreamHandler()

    def __init__(self, name, level=logging.INFO, tofile=None):
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

    def __call__(self, msg):
        self.logger.info(msg)    
