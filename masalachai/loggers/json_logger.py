# -*- coding: utf-8 -*-

import logging
import json
from pythonjsonlogger import jsonlogger
from masalachai.logger import Logger

class JsonLogger(Logger):

    formatter = jsonlogger.JsonFormatter()

    def __init__(self, name, logfile, level=logging.INFO):
        super(JsonLogger, self).__init__(name, level=level, logfile=logfile, train_log_mode='TRAIN_JSON', test_log_mode='TEST_JSON')

        self.mode['TRAIN_JSON'] = self.log_train_json_format
        self.mode['TEST_JSON'] = self.log_test_json_format


    def log_train_json_format(self, res):
        res['type'] = 'train'
        log_dict = {k: res[k] for k in res.keys()}
        return log_dict

    def log_test_json_format(self, res):
        res['type'] = 'test'
        log_dict = {k: res[k] for k in res.keys()}
        return log_dict
