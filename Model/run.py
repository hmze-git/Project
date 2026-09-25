import torch
import torchmetrics
import torchvision
import VAE
import RetinalDataLoader
from VAE import AnomalyVariationalAutoEncoder
from RetinalDataLoader import RetinalDiseaseLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.pyplot as plt

#CONFIG

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
INPUTCHANNELS=3
LATENTDIM=256
RDIM=32*7*7
LATENTSHAPE=(32,7,7)
BATCHSIZE=32

KLSCALEFIX=224*224*3 #divide by this so that the KL loss wont dominate when we use mean instead of sum 

LR=1e-3

BETA=0.5

NUMEPOCHS=300


#DATASET LOADER
trainDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\train(Processed)\Normal',True)
validationDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\val(Processed)\Normal',False)

trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)

#MODEL DEFINITION

AnomalyVAE=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)



#optimisers/early stopping/loss

lossFn=torch.nn.MSELoss(reduction="mean")

optimiser=Adam(AnomalyVAE.parameters(),lr=LR)




#TODEVICE
AnomalyVAE=AnomalyVAE.to(device=DEVICE)
seed = 67
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def trainingStep(epoch):
    epochRecon,epochKL,epochTotal=0.0,0.0,0.0
    AnomalyVAE.train()
    for batch,x in enumerate(trainSet):
        #reset the grad start each epoch so it does not accumulate
        
        optimiser.zero_grad()

        x=x.to(DEVICE)
   
        xRecon,mu,logvar=AnomalyVAE(x)

        reconLoss=lossFn(xRecon,x)

        klLoss=(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(),dim=1)).mean()

        klLoss=klLoss/KLSCALEFIX

        totalLoss=reconLoss+klLoss

        #backpropagate the loss 

        #do it with total so we can 'optmise' for both the current distribution and for centering around Gauss
        totalLoss.backward()

        optimiser.step()

        epochRecon+=reconLoss.item()
        epochKL+=klLoss.item()
        epochTotal+=totalLoss.item()

    
    if epoch%30==0:
        sampling(epoch=epoch)
    print(f" Training Epoch {epoch}->ReconLoss: {epochRecon} | KLLoss: {epochKL} | TotalLoss: {epochTotal}")

def validationStep(epoch):
    epochValRecon,epochValKL,epochValTotal=0.0,0.0,0.0
    for batch,x in enumerate(validSet):
        #Set to eval to eliminate the batch norm layers 
        AnomalyVAE.eval()

        with torch.no_grad():
            x=x.to(DEVICE)
        
            xRecon,mu,logvar=AnomalyVAE(x)
        
            reconValLoss=lossFn(xRecon,x)
        
            klValLoss=(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(),dim=1)).mean()

            klValLoss=klValLoss/KLSCALEFIX

            totalValLoss=reconValLoss+klValLoss
        
        
            epochValRecon+=reconValLoss.item()
            epochValKL+=klValLoss.item()
            epochValTotal+=totalValLoss.item()
    print(f" Validation Epoch {epoch}->ReconLoss: {epochValRecon} | KLLoss: {epochValKL} | TotalLoss: {epochValTotal}")        
    return epochValRecon,epochValKL,epochValTotal

def trainingLoop():
    for E in range(NUMEPOCHS):
        trainingStep(E)
        validationStep(E)

def sampling(epoch,numSave=5):
    AnomalyVAE.eval()
    with torch.no_grad():
        x=next(iter(trainSet))
        x=x.to(DEVICE)
       # z = torch.randn(BATCHSIZE, LATENTDIM)  # sample from prior
       # z=z.to(DEVICE)
        x_generated,mu,sigma = AnomalyVAE(x)  

        x_generated=x_generated.to('cpu')
        output=x_generated.view(-1,3,224,224)

        n=5
        fig,axes=plt.subplots(2,n,figsize=(15,16))

        x=x.cpu()
        output=output.cpu()
            
        for i in range(n):
            axes[0, i].imshow(x[i].permute(1, 2, 0))      # original
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            axes[1, i].imshow(output[i].permute(1, 2, 0))  # reconstruction
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig("recon_check.png")
        save_image(output,f"Generated{epoch}.png")



trainingLoop()
sampling(300)