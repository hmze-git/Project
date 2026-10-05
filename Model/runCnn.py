import torch
import torchmetrics
import torchvision
import VAE
import numpy as np
import math
import RetinalDataLoader
from CNN import AnomalyCNN
from RetinalDataLoader import RetinalDiseaseLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.utils import save_image
from torchmetrics import classification
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
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

#0.001
LR=1e-3

NUMEPOCHS=200


transformTrain=v2.Compose([
    v2.RandAugment(3,3),
    v2.ToTensor(),
])

transformValid=v2.Compose([
      v2.ToTensor(),
])

#DATASET LOADER
trainDatasetLoad=datasets.ImageFolder(r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\processCNN\train',transformTrain)
validationDatasetLoad=datasets.ImageFolder(r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\processCNN\validation',transformValid)

anomalyHoldoutsetLoad=datasets.ImageFolder(r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\testGuac(Processed)',transformValid)


trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)
anomalHolSet=DataLoader(anomalyHoldoutsetLoad,batch_size=BATCHSIZE,shuffle=True)



#MODEL DEFINITION
anomCNN=AnomalyCNN(inputChannels=INPUTCHANNELS,numCategories=3,rdim=RDIM)



#optimisers/early stopping/loss

lossFn=torch.nn.CrossEntropyLoss()
validLossFn=torch.nn.CrossEntropyLoss()


optimiser=Adam(anomCNN.parameters(),lr=LR)



#TODEVICE
anomCNN=anomCNN.to(device=DEVICE)
seed = 67
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#Metrics
accuracyMetric=classification.Accuracy(task='binary').to(DEVICE)
validAccuracyMetric=classification.Accuracy(task='binary').to(DEVICE)
precisionMetric=classification.Precision(task='binary').to(DEVICE)
recallMetric=classification.Recall(task='binary').to(DEVICE)

#Test Unseen Set Metrics
accMet=classification.Accuracy(task='binary').to(DEVICE)
precMet=classification.Precision(task='binary').to(DEVICE)
recMet=classification.Recall(task='binary').to(DEVICE)



def savingModel(Epoch,model,loss,optimiser,Path):

    torch.save({
        'modelStateDict':model.state_dict(),
        'loss':loss,
        'optimiserStateDict':optimiser.state_dict(),
        'epoch':Epoch
    },Path)
def plotChart(xLab,yLab,title,arr1,arr2,arrLegend,pathSave):

    

    plt.title(title)
    plt.plot(arr1)
    plt.plot(arr2)
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    plt.legend(arrLegend)
    plt.savefig(pathSave)
    plt.clf()


def trainStep(epoch):

    acummTrainLoss=0.0
    acummValidLoss=0.0

    anomCNN.train()

    for batch,(x,y) in enumerate(trainSet):
        optimiser.zero_grad()
        x=x.to(DEVICE)
        y=y.to(DEVICE)

        trainPreds=anomCNN(x)

        accuracyMetric.update(torch.argmax(trainPreds,dim=1),y)
        trainLoss=lossFn(trainPreds,y)

        acummTrainLoss+=trainLoss.item()

        trainLoss.backward()

        optimiser.step()

    anomCNN.eval()
    with torch.no_grad():

        for batch,(x,y) in enumerate(validSet):
            vX=x.to(DEVICE)
            vY=y.to(DEVICE)

            preds=anomCNN(vX)

            validAccuracyMetric.update(torch.argmax(preds,dim=1),vY)
            precisionMetric.update(torch.argmax(preds,dim=1),vY)
            recallMetric.update(torch.argmax(preds,dim=1),vY)

            validLoss=validLossFn(preds,vY)
            acummValidLoss+=validLoss.item()

    epochAccuracy=accuracyMetric.compute()
    epochValAccuracy=validAccuracyMetric.compute()
    epochPrecision=precisionMetric.compute()
    epochRecall=recallMetric.compute()
    acummTrainLoss=acummTrainLoss/len(trainSet)
    acummValidLoss=acummValidLoss/len(validSet)

    print(f"Epoch {epoch}| Loss {acummTrainLoss} | Validation Loss {acummValidLoss} | Accuracy {epochAccuracy} | Validation Accuracy {epochValAccuracy} | Precision {epochPrecision} | Recall {epochRecall} ")
    return epochAccuracy,epochValAccuracy,epochPrecision,epochRecall,acummTrainLoss,acummValidLoss 
def train():
    trainLoss=[]
    validLoss=[]
    trainAcc=[]
    validAcc=[]
    precision=[]
    recall=[]

    bestValLoss=float('inf')
    for E in range(NUMEPOCHS):
        epochAccuracy,epochValAccuracy,epochPrecision,epochRecall,acummTrainLoss,acummValidLoss =trainStep(E)


        trainLoss.append(acummTrainLoss)
        validLoss.append(acummValidLoss)
        trainAcc.append(epochAccuracy.cpu().item())
        validAcc.append(epochValAccuracy.cpu().item())
        precision.append(epochPrecision.cpu().item())
        recall.append(epochRecall.cpu().item())


        if acummValidLoss<bestValLoss:
            savingModel(E,anomCNN,validLoss,optimiser,'trainCNN.tar')
            bestValLoss=acummValidLoss

    plotChart('Epoch','Loss','CNN Train Vs Val Loss',trainLoss,validLoss,['train','val'],'TrainVsValLossCNN.png')
    plotChart('Epoch','Accuracy','CNN Train Vs Val Accuracy',trainAcc,validAcc,['train','val'],'TrainVsValACCCNN.png')
    plotChart('Epoch','Precision','CNN Precison Curve',precision,[],['precision'],'precision.png')
    plotChart("Epoch",'Recall',"CNN Recall Curve",recall,[],['Recall'],'recall.png')


def finalTest(modelFillePath,dataset,accuracy,precision,recall):

    if modelFillePath is not None:
            loadDict=torch.load(modelFillePath,map_location=DEVICE)
            anomCNN.load_state_dict(loadDict['modelStateDict'])
   

    yPreds=[]
    yTrue=[]
    for batch,(x,y) in enumerate(dataset):

    
        x=x.to(DEVICE)
        y=y.to(DEVICE)
   
        preds=anomCNN(x)
        indices=torch.argmax(preds,dim=1)
        yPreds.extend(indices.cpu().tolist())
        yTrue.extend(y.cpu().tolist())

        if accuracy is not None:
             #get the metric
            accuracy.update(indices,y)
            precision.update(indices,y)
            recall.update(indices,y)
            
            
            

    confMatrix=confusion_matrix(yTrue,yPreds,labels=[0,1])
    disp=ConfusionMatrixDisplay(confMatrix,display_labels=[0,1])
    disp.plot(cmap='Blues')
    plt.savefig('confMat.png')
    plt.show()
    plt.clf()


    if accuracy is not None:
        acc=accuracy.compute().item()
        rec=recall.compute().item()
        prec=precision.compute().item()
        print(f"Peak -> Accuracy:{acc} | Precision:{prec} | Recall:{rec}")
train()
#finalTest(None,validSet,None,None,None)
#inalTest('trainCNN.tar',anomalHolSet,accMet,precMet,recMet)