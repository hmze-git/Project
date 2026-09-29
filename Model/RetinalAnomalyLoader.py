import torch
import torchvision
from torch.utils.data import dataloader
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
import numpy as np
from PIL import Image

class RetinalDiseaseLoader(Dataset):


    def __init__(self,rootDir,trainMode=True):
        super().__init__()
        self.rootDir=rootDir
        self.images=os.listdir(self.rootDir)

        if trainMode==True:
            self.transform=v2.Compose([
                v2.RandomRotation(15),
                v2.ToTensor(),
            ])
        else:
            self.transform=v2.Compose([
                v2.ToTensor(),
            ])
        self.trainMode=trainMode

    def __getitem__(self, index):

        imagePath=os.path.join(self.rootDir,self.images[index])
        image=np.array(Image.open(imagePath))


        image=self.transform(image)
        return image

    def __len__(self):
        return len(self.images)