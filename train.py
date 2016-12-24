from __future__ import print_function
import argparse
import pickle
import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from vin import VIN

class MapData(chainer.dataset.DatasetMixin):
    def __init__(self, im, value, state, label):
        self.im = np.concatenate(
            (np.expand_dims(im, 1), np.expand_dims(value,1)),
            axis=1).astype(dtype=np.float32)
        self.s1, self.s2 = np.split(state, [1], axis=1)
        self.s1 = np.reshape(self.s1, self.s1.shape[0])
        self.s2 = np.reshape(self.s2, self.s2.shape[0])
        self.t = label.astype(np.int32)

    def __len__(self):
        return len(self.im)

    def get_example(self, i):
        return self.im[i], self.s1[i], self.s2[i], self.t[i]

def process_map_data(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)

    im_data = data['im']
    value_data = data['value']
    state_data = data['state']
    label_data = data['label']

    num = im_data.shape[0]
    num_train = num - num / 5

    im_train = im_data[0:num_train]
    value_train = value_data[0:num_train]
    state_train = state_data[0:num_train]
    label_train = label_data[0:num_train]

    im_test = im_data[num_train:-1]
    value_test = value_data[num_train:-1]
    state_test = state_data[num_train:-1]
    label_test = label_data[num_train:-1]

    train = MapData(im_train, value_train, state_train, label_train)
    test = MapData(im_test, value_test, state_test, label_test)

    return train, test

class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.predictor.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.predictor.train = True
        return ret

def main():
    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--data', '-d', type=str, default='./map_data.pkl',
                        help='Path to map data generated with script_make_data.py')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(VIN(k=20))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = process_map_data(args.data)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == "__main__":
    main()
