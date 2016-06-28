import chainer
import chainer.functions as F
import chainer.links as L

class Perceptrons(chainer.Chain):

    def __init__(self, in_size, hidden_size, activation=F.sigmoid):
        super(Perceptrons, self).__init__(
                fc1 = L.Linear(in_size, hidden_size),
        )
        self.activation = activation

    def __call__(self, x, train=True):
        return self.activation(self.fc1(x))

