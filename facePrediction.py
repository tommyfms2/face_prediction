
from image2TrainAndTest import image2TrainAndTest
from image2TrainAndTest import getValueData

import argparse
import numpy as np
from PIL import Image
import sys

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

def predict(model, valData):
    x = Variable(valData)
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]

    

def main():
    parse = argparse.ArgumentParser(description='face detection')
    parse.add_argument('--batchsize', '-b', type=int, default=100)
    parse.add_argument('--gpu', '-g', type=int, default=-1)
    parse.add_argument('--model','-m', default='')
    parse.add_argument('--optimizer', '-o', default='')
    parse.add_argument('--size', '-s', type=int, default=128)
    parse.add_argument('--channel', '-c', default=3)
    parse.add_argument('--testpath', '-p', default="./images/test/output/inputImage_0.png")
    args = parse.parse_args()

    errflag = -1
    if args.model == '':
        sys.stderr.write("Tom's Error occurred! ")
        sys.stderr.write("You have to designate the path to model")
        errflag = 1
    if args.optimizer == '':
        sys.stderr.write("Tom's Error occurred! ")
        sys.stderr.write("You have to designate the path to optimizer state")
        errflag = 1
    if errflag == 1:
        return

    outNumStr = args.model.split(".")[0].split("_")
    outnum = int(outNumStr[ len(outNumStr)-1 ])

    model = L.Classifier(Alex(args.channel, outnum))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    chainer.serializers.load_npz(args.model, model)
    chainer.serializers.load_npz(args.optimizer, optimizer)

    # fetch value data to predict who is he/she
    # getValueData is not created yet but provisionally done in image2trainandtest.py.
    valData = getValueData(args.testpath)
    pred = predict(model, valData)
    print(pred)

if __name__ == '__main__':
    main()

    
    
    
