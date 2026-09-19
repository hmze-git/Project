import torch
from torch import nn
from torch.distributions import normal

#Steps: Input->hiddenDim->mean,std->ReparamTrick->Decoder->OuputImage
class VariationalAutoEncoder(nn.Module):

    def __init__(self,inputDim,hiddenDim=200,zDim=20):
        super().__init__()

        #input layer to take 
        self.img2Hd=nn.Linear(inputDim,hiddenDim)
        self.hd2Mu=nn.Linear(hiddenDim,zDim)
        self.hd2Sigma=nn.Linear(hiddenDim,zDim)

        #decoder
        self.z2Hid=nn.Linear(zDim,hiddenDim)
        self.hid2Img=nn.Linear(hiddenDim,inputDim)

        self.relu=nn.ReLU() 

    def encode(self,x):
        h=self.relu(self.img2Hd(x))
        mu=self.hd2Mu(h)
        sigma=self.hd2Sigma(h)
        return mu,sigma

    def decode(self,z):
        h=self.relu(self.z2Hid(z))

        imgRet=torch.sigmoid(self.hid2Img(h))
        return imgRet

    def forward(self,x):
        mu,sigma=self.encode(x)
        epsilon=torch.randn_like(sigma)

        #epsiolon for intorducing stochasticity while removing it from netowrk so backprop can flow expectedly
        # and we dont have some random value that is unaccounted for that may be causing shifts and not being adjusted accordingly
        zReparam= mu+sigma*epsilon
        xRecontstruct=self.decode(zReparam)
        #Xrecon is for the MSE loss while mu and sigma is for the KL divergence(pushes to gaus)
        return xRecontstruct,mu,sigma

if __name__=="__main__":

    device=None
    if torch.cuda.is_available():
        device= torch.device('cuda')
        print(device)
    x=torch.randn(4,28*28)
    x=x.to(device)
    vae=VariationalAutoEncoder(inputDim=784)
    vae.to(device)
    xRecon,mu,sigma=vae(x)

    print(xRecon.shape)
    print(mu.shape)
    print(sigma.shape)
    print(vae(x))