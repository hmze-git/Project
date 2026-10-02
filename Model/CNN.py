import torch
import torchvision
from torch import nn

class AnomalyCNN(nn.Module):

    def __init__(self,inputChannels,numCategories,rdim=32*7*7):
        super().__init__()

        self.CNN=nn.Sequential(
            nn.Conv2d(in_channels=inputChannels,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(in_features=rdim,out_features=64),
            nn.Softmax(), #probabalistic output for last 2 neuros gives 
            nn.Linear(in_features=64,out_features=2)
        )

    def forward(self,X):
        xOut=self.CNN(X)


        return xOut