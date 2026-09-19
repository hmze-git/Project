from pathlib import Path
import cv2 as cv
import os
import matplotlib.pyplot as plt
DatasetPath=r"C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\dataset"

dimX=224
dimY=224

rootDir=Path(DatasetPath)

def processDataset():
    count=0
    for dir in rootDir.iterdir():

        for retType in dir.iterdir():
            os.makedirs(f"{str(dir.resolve())}(Processed)/{retType.name}",exist_ok=True)
            for retImg in retType.iterdir():

                    img = cv.imread(str(retImg.resolve()))

                    #Dont apply too may filters to produce images that can more closely mimic what will be seen 
                    #in reality
                    #come back if it doesnt perform well

                    resizeComp=cv.resize(img,(dimX,dimY))

                    clearedResize=cv.medianBlur(resizeComp,3)

                    cv.imwrite(f'{str(dir.resolve())}(Processed)/{retType.name}/{retImg.name}.png',clearedResize)


processDataset()