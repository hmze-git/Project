import torchvision
import torch
from torch import nn



class Encoder(nn.Module):

    #Expected size is 224x224
    def __init__(self,inputChannels=3):
        super().__init__()

        #Win: Input width
        #Hin: Input Heihgt
        #P: Padding amount
        #K: K kernel size
        #S: K Stride amount

        #WidthOut=((Win-K+2P)/S)+1
        #HeightOut=((Hin-K+2P)/S)+1

        self.Encode=nn.Sequential(
        #convLayers
        #NOTE:Conv layers in pytorch take number of channels in and how many out
        #How many out specifies how many filters to apply
        #NOTE: major padding not needed since we know images have black bordes
        nn.Conv2d(in_channels=3,out_channels=256,kernel_size=3,stride=2,padding=1),
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
        nn.Flatten()

        )

      
