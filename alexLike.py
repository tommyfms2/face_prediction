

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
from chainer import Chain, Variable
import chainer.links as L

class Alex(chainer.Chain):
    insize = 227
    def __init__(self, n_out):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, n_out),
        )
        self.train = True

    def __call__(self, x, t):
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
        
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

class AlexLike(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 128

    def __init__(self, n_out):
        super(AlexLike, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 1024),
            fc8=L.Linear(None, n_out),
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

class FromCaffeAlexnet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(FromCaffeAlexnet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=2),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            my_fc6=L.Linear(None, 4096),
            my_fc7=L.Linear(None, 1024),
            my_fc8=L.Linear(None, n_out),
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
        h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        h = self.my_fc8(h)
        return h

class GoogLeNet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(GoogLeNet, self).__init__(
            conv1=L.Convolution2D(None,  64, 7, stride=1, pad=3),
            conv2_reduce=L.Convolution2D(None,  64, 1),
            conv2=L.Convolution2D(None, 192, 3, stride=1, pad=1),
            inc3a=L.Inception(None,  64,  96, 128, 16,  32,  32),
            inc3b=L.Inception(None, 128, 128, 192, 32,  96,  64),
            inc4a=L.Inception(None, 192,  96, 208, 16,  48,  64),
            inc4b=L.Inception(None, 160, 112, 224, 24,  64,  64),
            inc4c=L.Inception(None, 128, 128, 256, 24,  64,  64),
            inc4d=L.Inception(None, 112, 144, 288, 32,  64,  64),
            inc4e=L.Inception(None, 256, 160, 320, 32, 128, 128),
            inc5a=L.Inception(None, 256, 160, 320, 32, 128, 128),
            inc5b=L.Inception(None, 384, 192, 384, 48, 128, 128),
            loss3_fc=L.Linear(None, n_out),
            
            loss1_conv=L.Convolution2D(None, 128, 1),
            loss1_fc1=L.Linear(None, 1024),
            loss1_fc2=L.Linear(None, n_out),
            
            loss2_conv=L.Convolution2D(None, 128, 1),
            loss2_fc1=L.Linear(None, 1024),
            loss2_fc2=L.Linear(None, n_out)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)
        
        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)
        
        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        loss1 = F.softmax_cross_entropy(l, t)
        
        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)
        
        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        loss2 = F.softmax_cross_entropy(l, t)
        
        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)
        
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4, train=self.train))
        loss3 = F.softmax_cross_entropy(h, t)
        
        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)
        
        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss
    
def predict(model, valData):
    x = Variable(valData)
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]

def predictNotPredictor(model, valData):
    x = Variable(valData)
    y = model.model(x)
    return y.data[0]


