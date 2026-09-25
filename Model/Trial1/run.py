import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from Model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim,nn


#config

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUTDIM=784
HDIM=200
ZDIM=64 # Compresses the latent space
LR=1e-4 #Karparthy constant
BATCHSIZE=32

NUMEPOCH=1000


#DATASER
#transforms.tensor normalises 
dataset=datasets.MNIST(root='dataset',train=True,transform=transforms.ToTensor(),download=True)
trainLoader=DataLoader(dataset,batch_size=BATCHSIZE,shuffle=True)
model=VariationalAutoEncoder(INPUTDIM,HDIM,ZDIM).to(DEVICE)
optimiser=optim.Adam(model.parameters(),lr=LR)
lossFn=nn.BCELoss(reduction="sum")

#start Train

for E in range(NUMEPOCH):
    loop=tqdm(trainLoader)

    for x,_ in loop:
    #forward

        x=x.to(DEVICE).view(x.shape[0],INPUTDIM)# Reshape to get input dim for Linear

        xRecon,mu,sigma=model(x)

        #push to recon imag
        reconLoss=lossFn(xRecon,x)

        #push towards std gause
        klDiv=-0.5*torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))



        #backprop
        loss=reconLoss+klDiv
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loop.set_postfix(loss=loss.item()/x.shape[0])


def inference(digit,numGen=5):
    images=[]
    idx=0
    for x,y in dataset:
        if y==idx:
            x=x.to(DEVICE)
            images.append(x)
            idx+=1

        if idx==10:
            break
    encodingDigit=[]
    for d in range(10):
        mu,sigman=model.encode(images[d].view(1,784))
        encodingDigit.append((mu,sigman))

    mu,sigma=encodingDigit[digit]

    for ex in range(numGen):
        epsilon=torch.randn_like(sigma)
        z=mu+sigma*epsilon
        out=model.decode(z)
        out=out.view(-1,1,28,28)
        save_image(out,f"Generated_{digit}_ex{ex}.png")


for idx in range(10):
    inference(idx,5)