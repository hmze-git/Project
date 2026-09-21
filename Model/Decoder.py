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

        #WidthOut=((Win-K+2P)/S)+1
        #HeightOut=((Hin-K+2P)/S)+1
        self.latenDim2Hidden=nn.Linear(in_features=latentDim,out_features=rdim)
       
        self.Decode=nn.Sequential(
        #convLayers
        #batch counts so dont use dim 0 or else it will break
        nn.Unflatten(dim=1,unflattened_size=shapeLatent),
            
        #Caclculate Channels*Height*width 
 
        )
        self.Mu=nn.Linear(in_features=32*7*7,out_features=latentDim)
        self.sigma=nn.Linear(in_features=32*7*7,out_features=latentDim)


    def forward(self,x):
        outFlat=self.Encode(x)

        #Both of these are now in the latent dim
        #return to autoencoder to apply epsion etc
        mu=self.mu(outFlat)
        sigma=self.sigma(outFlat)

        return mu,sigma

      
    #https://layercal.com/  helpful for visualisation of where params might bee too muchg