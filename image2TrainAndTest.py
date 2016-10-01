
import numpy as np
from PIL import Image
import glob
from chainer.datasets import tuple_dataset

def image2TrainAndTest(pathsAndLabels, size=128, channels=1):

    if channels == 1:
        imageData = []
        labelData = []
        for pathAndLabel in pathsAndLabels:
            path  = pathAndLabel[0]
            label = pathAndLabel[1]
            imagelist = glob.glob(path + "*")
            for imgName in imagelist:
                img = Image.open(imgName)
                imgData = np.asarray([np.float32(img)/255.0])
                #imgData2 = imgData.reshape(1, len(imgData)*len(imgData))[0]
                lblData = label
                imageData.append(imgData)
                labelData.append(np.int32(lblData))

        threshold = np.int32(len(imageData)/10*9)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])

    else:
        imageData = []
        labelData = []
        for pathAndLabel in pathsAndLabels:
            path  = pathAndLabel[0]
            label = pathAndLabel[1]
            imagelist = glob.glob(path + "*")
            for imgName in imagelist:
                img = Image.open(imgName)
                r,g,b = img.split()
                rImgData = np.asarray(np.float32(r)/255.0)
                gImgData = np.asarray(np.float32(g)/255.0)
                bImgData = np.asarray(np.float32(b)/255.0)
                imgData = np.asarray([rImgData, gImgData, bImgData])
                #imgData = np.asarray(np.float32(img)/255.0)
                #imgData2 = imgData.reshape(1, len(imgData)*len(imgData))[0]
                lblData = label
                imageData.append(imgData)
                labelData.append(np.int32(lblData))

        threshold = np.int32(len(imageData)/10*9)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        
    return train, test

def getValueDataFromPath(imagePath):
    img = Image.open(imagePath)
    img.show()
    r,g,b = img.split()
    rImgData = np.asarray(np.float32(r)/255.0)
    gImgData = np.asarray(np.float32(g)/255.0)
    bImgData = np.asarray(np.float32(b)/255.0)
    imgData = np.asarray([[[rImgData, gImgData, bImgData]]])
    return imgData

def getValueDataFromImg(img):
    img.show()
    r,g,b = img.split()
    rImgData = np.asarray(np.float32(r)/255.0)
    gImgData = np.asarray(np.float32(g)/255.0)
    bImgData = np.asarray(np.float32(b)/255.0)
    imgData = np.asarray([[[rImgData, gImgData, bImgData]]])
    return imgData

if __name__=='__main__':
    pathsAndLabels = []
    pathsAndLabels.append(np.asarray(["./images/akimoto/output/", 0]))
    pathsAndLabels.append(np.asarray(["./images/shiraishi/output/", 1]))
    train, test = image2TrainAndTest(pathsAndLabels)
    print(len(train))
    print(train[10])
    
