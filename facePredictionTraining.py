# -*- coding: utf-8 -*-

from image2TrainAndTest import image2TrainAndTest

import argparse
import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 128

    def __init__(self, input_channel, n_out):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(input_channel,  32, 8, stride=4),
            conv2=L.Convolution2D(32, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 256,  3, pad=1),
            conv4=L.Convolution2D(256, 256,  3, pad=1),
            conv5=L.Convolution2D(256, 32,  3, pad=1),
            fc6=L.Linear(288, 144),
            fc7=L.Linear(144, 50),
            fc8=L.Linear(50, n_out),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h


def main():
    parse = argparse.ArgumentParser(description='face detection train')
    parse.add_argument('--batchsize','-b',type=int, default=100,
                       help='Number if images in each mini batch')
    parse.add_argument('--epoch','-e',type=int, default=20,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                       help='GPU ID(negative value indicates CPU')
    parse.add_argument('--out','-o', default='result',
                       help='Directory to output the result')
    parse.add_argument('--resume','-r', default='',
                       help='Resume the training from snapshot')
    parse.add_argument('--unit','-u', type=int, default=1000,
                       help='Number of units')
    parse.add_argument('--model','-m', default='')
    parse.add_argument('--optimizer','-O', default='')
    parse.add_argument('--size','-s', type=int, default=128,
                       help='image size')
    parse.add_argument('--path','-p', default='')
    parse.add_argument('--channel','-c', default=3)

    args = parse.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    pathsAndLabels = []
    pathsAndLabels.append(np.asarray(["./images/the_others/output/", 0]))
    pathsAndLabels.append(np.asarray(["./images/akimoto/output/", 1]))
    pathsAndLabels.append(np.asarray(["./images/shiraishi/output/", 2]))
    pathsAndLabels.append(np.asarray(["./images/nishino/output/", 3]))
    pathsAndLabels.append(np.asarray(["./images/ikuta/output/", 4]))
    train, test = image2TrainAndTest(pathsAndLabels,channels=args.channel)
    

    model = L.Classifier(Alex(args.channel, len(pathsAndLabels)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    if args.model != '' and args.optimizer != '':
        chainer.serializers.load_npz(args.model, model)
        chainer.serializers.load_npz(args.optimizer, optimizer)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    outputname = "my_output_" + str(len(pathsAndLabels))
    modelOutName = outputname + ".model"
    OptimOutName = outputname + ".state"
    
    chainer.serializers.save_npz(modelOutName, model)
    chainer.serializers.save_npz(OptimOutName, optimizer)

if __name__ == '__main__':
    main()
