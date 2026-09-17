from pathlib import Path
import cv2
import os
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

                    img = cv2.imread(str(retImg.resolve()))
                
                    resizeImage=cv2.resize(img,(dimX,dimY))
                
                    cv2.imwrite(f'{retType.resolve()}(processed)/{retImg.name}.png',resizeImage)


processDataset()