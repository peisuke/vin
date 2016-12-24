from __future__ import print_function
import six
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class VIN(chainer.Chain):
    def __init__(self, k = 10, l_h=150, l_q=10, l_a=8):
        super(VIN, self).__init__(
            conv1=L.Convolution2D(2, l_h, 3, stride=1, pad=1),
            conv2=L.Convolution2D(l_h, 1, 1, stride=1, pad=0, nobias=True),

            conv3=L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),
            conv3b=L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),

            l3=L.Linear(l_q, l_a, nobias=True),
        )
        self.k = k
        self.train = True

    def __call__(self, x, s1, s2):
        h = F.relu(self.conv1(x))
        self.r = self.conv2(h)

        q = self.conv3(self.r)
        self.v = F.max(q, axis=1, keepdims=True)

        for i in xrange(self.k - 1):
            q = self.conv3(self.r) + self.conv3b(self.v)
            self.v = F.max(q, axis=1, keepdims=True)

        q = self.conv3(self.r) + self.conv3b(self.v)

        t = s2 * q.data.shape[3] + s1
        q = F.reshape(q, (q.data.shape[0], q.data.shape[1], -1))
        q = F.rollaxis(q, 2, 1)

        t_data_cpu = chainer.cuda.to_cpu(t.data)
        w = np.zeros(q.data.shape, dtype=np.float32)
        w[six.moves.range(t_data_cpu.size), t_data_cpu] = 1.0

        if isinstance(q.data, chainer.cuda.ndarray):
            w = chainer.cuda.to_gpu(w)

        w = chainer.Variable(w, volatile=not self.train)
        q_out = F.sum(w * q, axis=1)
        self.ret = self.l3(q_out)
        return self.ret


