
from PIL import Image
from PIL import ImageOps
import numpy as np

img = Image.open("mai.png")
#img = ImageOps.grayscale(imgC)
#img.show()
print(img.mode)
r,g,b = img.split()
rImgData = np.asarray(np.float32(r)/255.0)
gImgData = np.asarray(np.float32(g)/255.0)
bImgData = np.asarray(np.float32(b)/255.0)
imgDataOri = np.asarray(np.float32(img)/255.0)
#print(imgData)
imgData = np.asarray([rImgData, gImgData, bImgData])
#imgData.append(rImgData)
#imgData.append(gImgData)
#imgData.append(bImgData)
#print("ndim",imgData.ndim)

#h, w, k = imgData.shape
#hData = imgData[:,0,0]
#wData = imgData[0,:,0]
#kData = imgData[0,0,:]

#print(a)
#print(len(hData))
#print(len(wData))
#print(len(kData))
#print(kData)
#imgData2 = [[[None for col in range(750)] for row in range(499)] for k in range(3)]
#imgData2[:,0,0] = kData
#imgData2[0,:,0] = hData
#imgData2[0,0,:] = kData

print(imgData)
print(imgDataOri)

#理想
k, h, w = imgData.shape
print("k",k) #3
print("h",h) #750
print("w",w) #499

#実際
h, w, k = imgData.shape
print("h",h) #750
print("w",w) #499
print("k",k) #3

