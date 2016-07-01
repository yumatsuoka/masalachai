# -*- coding: utf-8 -*-

from chainer import link
from chainer.functions import identity

class Model(link.Chain):
    """ Abstract Model Class

    Model は，学習のための目的関数の計算と，推論過程の計算について定義を行うクラスです．

    Model のみを変えることによって同じネットワーク構造を記述したコードに対して，
    分類タスクや領域分割タスク，自己符号化などを実行することができます．

    Args:
        predictor (chainer.Chain): ネットワーク構造
        lossfun (function): 目的関数

    Attributes:
        lossfun (function): 目的関数
        y (chainer.Variable): 最後の入力バッチに対する推論結果
        loss (chainer.Variable): 最後の入力バッチに対する目的関数の出力
        accuracy (chainer.Variable): 最後の入力バッチに対する正解率
    """

    def __init__(self, predictor, lossfun=identity):
        super(Model, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, x, train=True):
        """ Computing loss value from an input

        入力から目的関数値を計算して，返します．

        Args:
            x (tuple of chainer.Variable): 入力値
            train (bool): 学習フェーズかどうか

        Returns:
            chainer.Variable: 目的関数の出力値
        """

        self.y = None
        self.loss = None
        x0, = x
        self.y = self.predictor(x0, train=train)
        self.loss = self.lossfun(self.y)
        return self.loss

    def predict(self, x):
        """Computing loss value from an input

        入力から推論を実行して，結果を返します．

        Args:
            x (tuple of chainer.Variable): 入力値
            train (bool): 学習フェーズかどうか

        Returns:
            chainer.Variable: 推論結果
        """

        x0, = x
        self.y = self.predictor(x0, train=False)
        return self.y
