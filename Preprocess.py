from pathlib import Path
import cv2 as cv
import os
import matplotlib.pyplot as plt
DatasetPath=r"C:\Users\Hamzah\Desktop\HYP\AAI\archive\dataset"

dimX=224
dimY=224

rootDir=Path(DatasetPath)

def processDataset():
    count=0
    for dir in rootDir.iterdir():
        for retType in dir.iterdir():
            os.makedirs(f"{str(retType.resolve())}(processed)",exist_ok=True)
            for retImg in retType.iterdir():

                    img = cv.imread(str(retImg.resolve()))


                    resizeComp=cv.resize(img,(dimX,dimY))

                    #Greyscale to threshold with a min max range
                    grey=cv.cvtColor(img,cv.COLOR_RGB2GRAY)

                    _,mask=cv.threshold(grey,10,255,cv.THRESH_BINARY)

                    #usually do bitwise and with 2 images doing with same img gives img so pointless 
                    #but needed so it wont break
                    #mask does heavy lifting
                    maskedImage=cv.bitwise_and(img,img,mask=mask)

                    maskedImage=cv.medianBlur(maskedImage,3)

                    resizedMask=cv.resize(maskedImage,(dimX,dimY))
                    resizeImage=cv.resize(img,(dimX,dimY))

                    
                    plt.figure()
                    plt.subplot(121)
                    plt.imshow(resizedMask)
                    plt.subplot(122)
                    plt.imshow(resizeComp)

                    plt.show()
    
                    #cv.imwrite(f'{retType.resolve()}(processed)/{retImg.name}.png',resizeImage)


processDataset()