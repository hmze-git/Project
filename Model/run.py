import torch
import torchmetrics
import torchvision
import VAE
import numpy as np
import math
import RetinalDataLoader
from VAE import AnomalyVariationalAutoEncoder
from RetinalDataLoader import RetinalDiseaseLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import v2 
import matplotlib.pyplot as plt

#CONFIG

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
INPUTCHANNELS=3
LATENTDIM=64
RDIM=32*7*7
LATENTSHAPE=(32,7,7)
BATCHSIZE=32

RECONSCALEFIX=224*224*3 #divide by this so that the KL loss wont dominate when we use mean instead of sum 

#0.001
LR=1e-3


BETA=20
NUMEPOCHS=210


#DATASET LOADER
trainDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\train(Processed)\Normal',True)
validationDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\val(Processed)\Normal',False)

transform=v2.Compose([
    v2.RandomRotation(15),
    v2.ToTensor(),
])

anomalyDatasetLoad=datasets.ImageFolder(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\trainAnomaly',transform)




trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)
anomalyset=DataLoader(anomalyDatasetLoad,batch_size=BATCHSIZE,shuffle=True)

#MODEL DEFINITION

AnomalyVAE=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)



#optimisers/early stopping/loss

lossFn=torch.nn.MSELoss(reduction="sum")


optimiser=Adam(AnomalyVAE.parameters(),lr=LR)

lrScheduler=ReduceLROnPlateau(optimizer=optimiser,mode='min',patience=40,factor=0.5)



#TODEVICE
AnomalyVAE=AnomalyVAE.to(device=DEVICE)
seed = 67
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def sigmoidKlAnnealing(timeStep,slope=1,start=None):
    return float(1/(1+np.exp(slope*(start-float(timeStep)))))
    

def anomalyCheck(epoch):
    for b,(x,y) in enumerate(anomalyset):
        x,y=x.to(DEVICE),y.to(DEVICE)

def savingModel(Epoch,model,loss,optimiser,Path):

    torch.save({
        'modelStateDict':model.state_dict(),
        'loss':loss,
        'optimiserStateDict':optimiser.state_dict(),
        'epoch':Epoch
    },Path)




def trainingStep(epoch,BETA):
    epochRecon,epochKL,epochTotal=0.0,0.0,0.0
    AnomalyVAE.train()
    
    for batch,x in enumerate(trainSet):
        #reset the grad start each epoch so it does not accumulate
        
        optimiser.zero_grad()

        x=x.to(DEVICE)
   
        xRecon,mu,logvar=AnomalyVAE(x)

        reconLoss=lossFn(xRecon,x)

        klLoss=(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))


        #Fix the scale of the losses before backprop and sumation
        reconLoss=reconLoss/RECONSCALEFIX

        klLoss=klLoss/RECONSCALEFIX

        totalLoss=reconLoss+BETA*klLoss
     
   

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
    return epochRecon,epochKL,epochTotal

def validationStep(epoch,BETA):
    epochValRecon,epochValKL,epochValTotal=0.0,0.0,0.0

    for batch,x in enumerate(validSet):
        #Set to eval to eliminate the batch norm layers 
        AnomalyVAE.eval()

        with torch.no_grad():
            x=x.to(DEVICE)
       
         
            xRecon,mu,logvar=AnomalyVAE(x)
        
            reconValLoss=lossFn(xRecon,x)
        
            klValLoss=(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()))


            #Scale the loss accordingly

            reconValLoss=reconValLoss/RECONSCALEFIX

            klValLoss=klValLoss/RECONSCALEFIX

            totalValLoss=reconValLoss+BETA*klValLoss
        

            #Lear Rate Scheduler
            #lrScheduler.step(totalValLoss)

            epochValRecon+=reconValLoss.item()
            epochValKL+=klValLoss.item()
            epochValTotal+=totalValLoss.item()
    print(f" Validation Epoch {epoch}->ReconLoss: {epochValRecon} | KLLoss: {epochValKL} | TotalLoss: {epochValTotal}")        
    return epochValRecon,epochValKL,epochValTotal

def plotChart(xLab,yLab,title,arr1,arr2,arrLegend,pathSave):

    plt.title(title)
    plt.plot(arr1)
    plt.plot(arr2)
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    plt.legend(arrLegend)
    plt.savefig(pathSave)
    plt.clf()



def trainingLoop():
    #This ensures that beta slowly trends upwards from 0 so we can start with more focus on recon loss and then after 100 epochs we can then focus more on the 
    #KL anealing
    trainReconL=[]
    trainKLL=[]
    trainTotalL=[]
    validReconL=[]
    validKLL=[]
    validTotalL=[]


    bestValidationLoss=float('inf') # absolute biggest loss value possible
    for E in range(NUMEPOCHS):
       ## BETA=sigmoidKlAnnealing(E,0.1,100)
      ##  BETA=min(BETACLAMP,BETA) #clamp it because when beta gets close to 1 alot of reconstruction accuracy is lost
        
        print(BETA)
        trRec,trKL,trTot=trainingStep(E,BETA)
        vlRec,vlKL,vlTot=validationStep(E,BETA)

        #append the values to array to use in charts later
        trainReconL.append(trRec)
        trainKLL.append(trKL)
        trainTotalL.append(trTot)
        #validation section
        validReconL.append(vlRec)
        validKLL.append(vlKL)
        validTotalL.append(vlTot)

        if vlTot<bestValidationLoss:
            savingModel(E,AnomalyVAE,vlTot,optimiser,'bestModel.tar')
            bestValidationLoss=vlTot

    plotChart('Epoch','Reconstruction Loss','VAE Recon Loss',trainReconL,validReconL,['train','val'],'trainVSvalRecon.png')
    plotChart('Epoch','KL Loss','VAE KL Loss',trainKLL,validKLL,['train','val'],'trainVSvalKLL.png')
    plotChart('Epoch','Total Loss','VAE Total Loss',trainTotalL,validTotalL,['train','val'],'trainVSvalTOT.png')

def anomalyCheck(E):
    pass



def sampling(epoch,numSave=5):
    AnomalyVAE.eval()
    with torch.no_grad():
        x=next(iter(trainSet))
        x=x.to(DEVICE)
        z = torch.randn(BATCHSIZE, LATENTDIM)  # sample from gaus normal and see if we can get good looking recons
        z=z.to(DEVICE)
        x_generated,mu,sigma = AnomalyVAE(x) 
 

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
        plt.clf()
        save_image(output,f"Generated{epoch}.png")

        randomSample=AnomalyVAE.decoder(z)


        randomSample=randomSample.view(-1,3,224,224)
        xRanRec=randomSample.to('cpu')
        save_image(xRanRec,f"GeneratedRandom{epoch}.png")



trainingLoop()
sampling(210)