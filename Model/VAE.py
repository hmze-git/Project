import torch
import torchvision
from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class AnomalyVariationalAutoEncoder(nn.Module):


    def __init__(self,inputChannels=3,latenDim=64,rdim=(32*7*7),shapeLatent=(32,7,7)):
        super().__init__()
        self.encoder=Encoder(inputChannels=inputChannels,latentDim=latenDim,rdim=rdim)
        self.decoder=Decoder(inputChannels=inputChannels,latentDim=latenDim,rdim=rdim,shapeLatent=shapeLatent)

    def encoding(self,x):
        zParam,mu,logvar=self.encoder(x)
        return zParam,mu,logvar
        

    def decoding(self,zReparam):
        xReconstructed=self.decoder(zReparam) 

        return xReconstructed

    def forward(self,x):

        zReparam,mu,logVar=self.encoding(x)

        xRecon=self.decoding(zReparam)

        return xRecon,mu,logVar

    
    