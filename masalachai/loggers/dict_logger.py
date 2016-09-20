# -*- coding: utf-8 -*-

import os
import logging
from masalachai.logger import Logger


class DictLogger(Logger):

    def __init__(self, name, logfile, level=logging.INFO):
        super(DictLogger, self).__init__(
            name, level=level, logfile=None,
            train_log_mode='TRAIN_DICT', test_log_mode='TEST_DICT')
        self.output_file = logfile
        self.log_data = []

        self.mode['TRAIN_DICT'] = self.log_train_dict_format
        self.mode['TEST_DICT'] = self.log_test_dict_format

    def __call__(self, msg):
        if isinstance(msg, tuple):
            m, d = msg
        else:
            m = msg
            d = {'massage': m}
        self.log_data.append(d)
        self._logger.info(m)

    def post_log(self):
        if (len(os.path.dirname(self.output_file)) > 0) and (
                not os.path.exists(os.path.dirname(self.output_file))):
            os.makedirs(os.path.dirname(self.output_file))

        with open(self.output_file, 'w') as f:
            for d in self.log_data:
                f.write('%s\n' % str(d))

    def log_train_dict_format(self, res):
        res['type'] = 'train'
        log_dict = {k: res[k] for k in res.keys()}
        log_str = '{0:d}, loss={1:.5f}, accuracy={2:.5f}'.format(
            res['iteration'], res['loss'], res['accuracy'])
        return log_str, log_dict

    def log_test_dict_format(self, res):
        res['type'] = 'test'
        log_dict = {k: res[k] for k in res.keys()}
        log_str = '[TEST], loss={0:.5f}, accuracy={1:.5f}'.format(
            res['loss'], res['accuracy'])
        return log_str, log_dict
