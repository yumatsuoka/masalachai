# -*- coding: utf-8 -*-

import os
import logging
import threading

class Logger(threading.Thread):
    """ Logger Class

    Logger は，学習時の目的関数値や正解率などを出力するクラスです．

    このクラスは標準ライブラリモジュール`logging <http://docs.python.jp/3/library/logging.html>`_ をラップしているため，
    このクラスでは，loggingモジュールの機能を利用する事ができます．

    Args:
        name (str): ロガー名．ログを取る際のnameフィールドに挿入される文字列です．
        level (): 
        logfile (str): logfileが設定されると，設定されたファイル名でログを保存します．
        train_log_mode (str): 学習時のログモードを指定します．デフォルトで利用できるモードは'TRAIN'と'TRAIN_LOSS_ONLY'で，'TRAIN'は目的関数値と正解率を，'TRAIN_LOSS_ONLY'は目的関数値のみを出力します．
        test_log_mode (str): テスト時のログモードを指定します．デフォルトで利用できるモードは'TEST'と'TEST_LOSS_ONLY'で，'TEST'は目的関数値と正解率を，'TEST_LOSS_ONLY'は目的関数値のみを出力します．

    Attributes:
        mode (dict): ログモード名をkey，ログ関数をvalueにもった辞書です．modeにkeyとvalueを追加することで好きなフォーマットのログモードを追加できます．
        train_log_mode (str): 学習時のログモード名です．
        test_log_mode (str): テスト時のログモード名です．
        queue (queue.Queue): マルチスレッディング動作時に~Trainerとデータをやり取りするためのキューです．
        stop (threading.Event): マルチスレッディング動作時に~Trainerから学習終了シグナルをやり取りするためのイベントです．
    """

    formatter = logging.Formatter(fmt='%(asctime)s %(name)8s %(levelname)8s: %(message)s',datefmt='%Y/%m/%d %p %I:%M:%S,',)

    def __init__(self, name, level=logging.INFO, logfile=None, train_log_mode='TRAIN', test_log_mode='TEST'):
        super(Logger, self).__init__()
        self.setDaemon(True)

        # stream handler setting
        self._handler = logging.StreamHandler()
        self._handler.setLevel(level)
        self._handler.setFormatter(self.formatter)

        # logger setting
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)

        # file handler setting
        if logfile is not None:
            self._file_handler = logging.FileHandler(logfile, mode='w')
            self._file_handler.setFormatter(self.formatter)
            self._logger.addHandler(self._file_handler)
        
        self.mode = {'TRAIN': self.log_train, 'TRAIN_LOSS_ONLY': self.log_train_loss_only, 'TEST': self.log_test, 'TEST_LOSS_ONLY': self.log_test_loss_only, 'END': None}
        self.train_log_mode = train_log_mode
        self.test_log_mode = test_log_mode
        self.queue = None
        self.stop = None


    def __call__(self, msg):
        """ Logging function

        入力された内容をロギングハンドラへ渡して設定された出力へログを表示します．

        Args:
            msg (str): ログ内容
        """

        self._logger.info(msg)


    def setQueue(self, queue):
        """ Setter of my queue

        キューオブジェクトを登録します．

        Args:
            queue (queue.Queue): 登録したいキューオブジェクト

        """

        self.queue = queue

    def post_log(self):
        """ Post-process for end of thread

        ログスレッドが終了するときに必要な処理を行います．
        Loggerクラス上では，特に何も行われません．

        """
        pass


    def run(self):
        """ Running logging thread

        ログスレッドを走らせます．

        .. note::
            このクラスは`threading.Thread <http://docs.python.jp/3/library/threading.html#thread-objects>`_ のサブクラスです．
            ログスレッドを走らせるには，start() メソッドを呼び出してください．

            ログスレッドを終了させるには，self.stop.set()を呼び出すか，
            親スレッドを終了させてください（ログスレッドはデーモンスレッドとして走ります）．

            また，一度ストップさせたスレッドは再開させることが出来ないことに注意してください．
        """

        # queue check
        assert self.queue is not None, "Log Queue is None, use Logger.setQueue(queue) before calling me."

        self.stop = threading.Event()
        while not self.stop.is_set():
            res = self.queue.get()
            if getattr(res,'__hash__',False) and res in self.mode:
                log_func = self.mode[res]
                if res == 'END':
                    self.stop.set()
                continue
            self.__call__(log_func(res))

        self.post_log()


    def log_train(self, res):
        """ Logging formatter for training

        学習時のログ内容をフォーマットして出力します．

        Args:
            res (dict): ログをとりたい情報を持った辞書です．学習回数を表す'iteration'，目的関数値を表す'loss'，正解率を表す'accuracy'のそれぞれのkeyを持つ辞書である必要があります．

        Returns:
            str: ログのためににフォーマットされた文字列
        """

        log_str = '{0:d}, loss={1:.5f}, accuracy={2:.5f}'.format(res['iteration'], res['loss'], res['accuracy'])
        return log_str

    def log_train_loss_only(self, res):
        """ Logging formatter for training (LOSS ONLY)

        学習時のログ内容（正解率出力なし）をフォーマットして出力します．

        Args:
            res (dict): ログをとりたい情報を持った辞書です．学習回数を表す'iteration'，目的関数値を表す'loss'のそれぞれのkeyを持つ辞書である必要があります．

        Returns:
            str: ログのためににフォーマットされた文字列
        """

        log_str = '{0:d}, loss={1:.5f}'.format(res['iteration'], res['loss'])
        return log_str

    def log_test(self, res):
        """ Logging formatter for testing

        テスト時のログ内容をフォーマットして出力します．

        Args:
            res (dict): ログをとりたい情報を持った辞書です．目的関数値を表す'loss'，正解率を表す'accuracy'のそれぞれのkeyを持つ辞書である必要があります．

        Returns:
            str: ログのためににフォーマットされた文字列
        """

        log_str = '[TEST], loss={0:.5f}, accuracy={1:.5f}'.format(res['loss'], res['accuracy'])
        return log_str

    def log_test_loss_only(self, res):
        """ Logging formatter for testing (LOSS ONLY)

        学習時のログ内容をフォーマットして出力します．

        Args:
            res (dict): ログをとりたい情報を持った辞書です．目的関数値を表す'loss'をkeyに持つ辞書である必要があります．

        Returns:
            str: ログのためににフォーマットされた文字列
        """

        log_str = '[TEST], loss={0:.5f}'.format(res['loss'])
        return log_str


