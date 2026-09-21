import torchvision
import torch
from torch import nn



class Decoder(nn.Module):

    #Expected size is 224x224
    def __init__(self,inputChannels=3,latentDim=64,rdim=32*7*7,shapeLatent=(32,7,7)):
        super().__init__()

        #Win: Input width
        #Hin: Input Heihgt
        #P: Padding amount
        #K: K kernel size
        #S: K Stride amount
        self.channel,self.width,self.height=shapeLatent
        #WidthOut=((Win-K+2P)/S)+1
        #HeightOut=((Hin-K+2P)/S)+1
        self.latenDim2Hidden=nn.Linear(in_features=latentDim,out_features=rdim)
       
        self.Decode=nn.Sequential(
        #convLayers
        #batch counts so dont use dim 0 or else it will break
        nn.Unflatten(dim=1,unflattened_size=shapeLatent),
        nn.ConvTranspose2d(in_channels=self.channel,out_channels=64,kernel_size=3,stride=2),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=3,stride=2),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=2),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=128,out_channels=256,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=256),
        nn.ConvTranspose2d(in_channels=256,out_channels=inputChannels,kernel_size=3,stride=2),
        nn.BatchNorm2d(num_features=inputChannels),
        nn.Sigmoid(),

        #Caclculate Channels*Height*width 
 
        )
        self.Mu=nn.Linear(in_features=32*7*7,out_features=latentDim)
        self.sigma=nn.Linear(in_features=32*7*7,out_features=latentDim)


    def forward(self,x):
        sigmoidOut=self.Decode(x)
        return sigmoidOut

      
    #https://layercal.com/  helpful for visualisation of where params might bee too muchg