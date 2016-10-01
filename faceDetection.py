
import cv2
from PIL import Image
import numpy as np
import argparse
import glob

def faceDetectionWithPath(path, size):
    cvImg = cv2.imread(path)
    cascade_path = "./lib/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    color = (255, 255, 255)
    faceData = []
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, fx=float(size/faceImg.shape[0]),fy=float( size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg)
        
    return faceData

def faceDetectionWithPil(img, size):
    cvImg = np.asarray(img)
    cascade_path = "./lib/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    color = (255, 255, 255)
    faceData = []
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, fx=float(size/faceImg.shape[0]),fy=float( size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg)
        
    return faceData

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="image path")
    parse.add_argument('--path', '-p', default="")
    parse.add_argument('--output','-o', default="./images/test/output/")
    parse.add_argument('--size', '-s', type=float, default=128)
    args = parse.parse_args()
    
    if args.path != "":
        faceImg = faceDetectionWithPath(args.path, args.size)
        print(faceImg)
        imagelist = glob.glob(args.output + "*")
        counter = 1
        for img in faceImg:
            img.save(args.output + "output_" + str(len(imagelist)+counter) + ".png")
            counter = counter + 1
    
