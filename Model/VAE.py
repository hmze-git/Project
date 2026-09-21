import torch
import torchvision
from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class AnomalyVariationalAutoEncoder(nn.Module):


    def __init__(self,inputChannels=3,latenDim=64,rdim=(32*7*7)):
        super().__init__()
        self.encoder=Encoder(inputChannels=inputChannels,latentDim=latenDim,rdim=rdim)
        self.decoder=Decoder(inputChannels=inputChannels,latentDim=latenDim,rdim=rdim,shapeLatent=(32,7,7))

    def encoder(self,x):
        mu,sigma=self.encoder(x)
   

    def decoder(self,z):
        pass

    
    