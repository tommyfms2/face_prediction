# -*- coding: utf-8 -*-

from image2TrainAndTest import image2TrainAndTest
import alexLike

import argparse
import numpy as np
from PIL import Image
import glob

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions

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
    label_i = 0
    data_list = glob.glob(args.path + "*")
    print(data_list)
    for datafinderName in data_list:
        pathsAndLabels.append(np.asarray([datafinderName+"/", label_i]))
        label_i = label_i + 1

    train, test = image2TrainAndTest(pathsAndLabels,channels=args.channel)
    

    model = L.Classifier(alexLike.AlexLike( len(pathsAndLabels)))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU


    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if args.model != '' and args.optimizer != '':
        chainer.serializers.load_npz(args.model, model)
        chainer.serializers.load_npz(args.optimizer, optimizer)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
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
