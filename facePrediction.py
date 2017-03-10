
from image2TrainAndTest import image2TrainAndTest
from image2TrainAndTest import getValueDataFromPath
from image2TrainAndTest import getValueDataFromImg
from faceDetection import faceDetectionFromPath
import alexLike

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
    

def main():
    parse = argparse.ArgumentParser(description='face detection')
    parse.add_argument('--batchsize', '-b', type=int, default=100)
    parse.add_argument('--gpu', '-g', type=int, default=-1)
    parse.add_argument('--model','-m', default='')
    parse.add_argument('--size', '-s', type=int, default=128)
    parse.add_argument('--channel', '-c', default=3)
    parse.add_argument('--testpath', '-p', default="./images/test/output/inputImage_0.png")
    args = parse.parse_args()

    if args.model == '':
        sys.stderr.write("Tom's Error occurred! ")
        sys.stderr.write("You have to designate the path to model")
        return

    outNumStr = args.model.split(".")[0].split("_")
    outnum = int(outNumStr[ len(outNumStr)-1 ])

    model = L.Classifier(alexLike.AlexLike(outnum))
    chainer.serializers.load_npz(args.model, model)

    ident = [""] * outnum
    for line in open("whoiswho.txt", "r"):
        dirname = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dirname

    # fetch value data to predict who is he/she
    faceImgs = faceDetectionFromPath(args.testpath, args.size)
    for faceImg in faceImgs:
        valData = getValueDataFromImg(faceImg)
        pred = alexLike.predict(model, valData)
        print(pred)
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                print("he/she is {}".format(ident[pre_i]))
            

if __name__ == '__main__':
    main()

    
    
    
