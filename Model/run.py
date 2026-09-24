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

#CONFIG

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
INPUTCHANNELS=3
LATENTDIM=64
RDIM=32*7*7
LATENTSHAPE=(32,7,7)
BATCHSIZE=32

LR=1e-4

BETA=0.5

NUMEPOCHS=1000


#DATASET LOADER
trainDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\train(Processed)\Normal',True)
validationDatasetLoad=RetinalDiseaseLoader(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\val(Processed)\Normal')

trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)

#MODEL DEFINITION

AnomalyVAE=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)



#optimisers/early stopping/loss

lossFn=torch.nn.BCELoss(reduction='sum')

optimiser=Adam(AnomalyVAE.parameters(),lr=LR)




#TODEVICE
AnomalyVAE=AnomalyVAE.to(device=DEVICE)
seed = 67
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def trainingStep(epoch):
    for batch,x in enumerate(trainSet):
        #reset the grad start each epoch so it does not accumulate
        optimiser.zero_grad()

        x=x.to(DEVICE)
   
        xRecon,mu,logvar=AnomalyVAE(x)

        reconLoss=lossFn(xRecon,x)

        klLoss=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())

        totalLoss=reconLoss+klLoss


        #backpropagate the loss 

        #do it with total so we can 'optmise' for both the current distribution and for centering around Gauss
        totalLoss.backward()

        optimiser.step()

    print(f" Training Epoch {epoch}->ReconLoss: {reconLoss} | KLLoss: {klLoss} | TotalLoss: {totalLoss}")

def trainingLoop():
    for E in range(NUMEPOCHS):
        trainingStep(E)

def sampling(numSave=5):
    AnomalyVAE.eval()
    with torch.no_grad():
        z = torch.randn(BATCHSIZE, LATENTDIM)  # sample from prior
        z=z.to(DEVICE)
        x_generated = AnomalyVAE.decoding(z)     

        x_generated=x_generated.to('cpu')
        output=x_generated.view(-1,3,224,224)
        save_image(output,f"Generated1.png")



trainingLoop()
sampling()