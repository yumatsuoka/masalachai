import chainer
import chainer.functions as F
import chainer.links as L

class Mlp(chainer.Chain):
    
    def __init__(self, in_size, hidden1_size, hidden2_size, out_size):
        super(Mlp, self).__init__(
                fc1 = L.Linear(in_size, hidden1_size),
                bn1 = L.BatchNormalization(hidden1_size),
                fc2 = L.Linear(hidden1_size, hidden2_size),
                bn2 = L.BatchNormalization(hidden2_size),
                fc3 = L.Linear(hidden2_size, out_size),
        )
        self.train = True

    def __call__(self, x, t=None):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)
