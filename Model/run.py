import torch 
import torchmetrics
import torchvision
from Model.VAE import AnomalyVariationalAutoEncoder


#CONFIG

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUTCHANNELS=3
LATENTDIM=64
RDIM=32*7*7
LATENTSHAPE=(32,7,7)
BATCHSIZE=16
LR=1e-4

NUMEPOCHS=50



#MODEL DEFINITION

AnomalyVAE=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)