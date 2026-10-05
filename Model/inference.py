import torch
import torchmetrics
import torchvision
import VAE
import numpy as np
from PIL import Image
import math
import RetinalDataLoader
from VAE import AnomalyVariationalAutoEncoder
from RetinalDataLoader import RetinalDiseaseLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.utils import save_image
from torchmetrics import classification
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score
from torchvision.transforms import v2 
from CNN import AnomalyCNN
import matplotlib.pyplot as plt
from matplotlib import image as img
import tkinter as tk
from tkinter import filedialog
# Settings


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
NUMEPOCHS=200


#DATASET PATHS

VAETRAIN=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\process2\train(Processed)\Normal'
VAETEST=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\process2\val(Processed)\Normal'
VAEANOMALYTEST=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\process2\trainAnomaly'
VAEGLUACOMATEST=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\testGuac(Processed)'



CNNTRAIN=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\processCNN\train'
CNNTEST=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\processCNN\validation'
CNNGLUACOMATEST=r'C:\Users\Hamzah\Desktop\HYP\Dataset\AI\testGuac(Processed)'



#BEST MODELS
VAEBEST='bestModel.tar'
CNNBEST='trainCNN.tar'

seed = 67
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
## COMMON METHODS
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
## END COMMON


#CNNN OPNLY
def trainStepCNN(epoch,anomCNN,trainSet,validSet,lossFn,validLossFn,optimiser,accuracyMetric,validAccuracyMetric,precisionMetric,recallMetric):

    acummTrainLoss=0.0
    acummValidLoss=0.0



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
def runTrainingCNN(anomCNN,trainSet,validSet,lossFN,validLossFN,optimiser,accuracyMetric,validAccuracyMetric,precisionMetric,recallMetric):
    trainLoss=[]
    validLoss=[]
    trainAcc=[]
    validAcc=[]
    precision=[]
    recall=[]

    bestValAccuracy=float('inf')
    for E in range(NUMEPOCHS):
        epochAccuracy,epochValAccuracy,epochPrecision,epochRecall,acummTrainLoss,acummValidLoss =trainStepCNN(E,anomCNN,trainSet,validSet,lossFN,validLossFN,optimiser,accuracyMetric,validAccuracyMetric,precisionMetric,recallMetric)


        trainLoss.append(acummTrainLoss)
        validLoss.append(acummValidLoss)
        trainAcc.append(epochAccuracy.cpu().item())
        validAcc.append(epochValAccuracy.cpu().item())
        precision.append(epochPrecision.cpu().item())
        recall.append(epochRecall.cpu().item())


        if epochValAccuracy<bestValAccuracy:
            savingModel(E,anomCNN,validLoss,optimiser,'trainCNN.tar')
            bestValAccuracy=epochValAccuracy

    plotChart('Epoch','Loss','CNN Train Vs Val Loss',trainLoss,validLoss,['train','val'],'TrainVsValLossCNN.png')
    plotChart('Epoch','Accuracy','CNN Train Vs Val Accuracy',trainAcc,validAcc,['train','val'],'TrainVsValACCCNN.png')
    plotChart('Epoch','Precision','CNN Precison Curve',precision,[],['precision'],'precision.png')
    plotChart("Epoch",'Recall',"CNN Recall Curve",recall,[],['Recall'],'recall.png')

def finalTestCNN(model,modelFillePath,dataset,accuracy,precision,recall):

    if modelFillePath is not None:
            loadDict=torch.load(modelFillePath,map_location=DEVICE)
            model.load_state_dict(loadDict['modelStateDict'])
   

    yPreds=[]
    yTrue=[]
    for batch,(x,y) in enumerate(dataset):

    
        x=x.to(DEVICE)
        y=y.to(DEVICE)
   
        preds=model(x)
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

## END CNN ONlY


## VAE METHODS

def sampling(epoch,BETAVAE,trainSet):
    BETAVAE.eval()
    with torch.no_grad():
        x=next(iter(trainSet))
        x=x.to(DEVICE)
        z = torch.randn(BATCHSIZE, LATENTDIM)  # sample from gaus normal and see if we can get good looking recons
        z=z.to(DEVICE)
        xGenerated,mu,sigma = BETAVAE(x) 
 

        output=xGenerated.view(-1,3,224,224)

        save_image(output,f"Generated{epoch}.png")

        randomSample=BETAVAE.decoder(z)


        randomSample=randomSample.view(-1,3,224,224)
        xRanRec=randomSample.to('cpu')
        save_image(xRanRec,f"GeneratedRandom{epoch}.png")

def trainingStep(epoch,BETA,trainSet,VAE,optimiser,lossFN):
    epochRecon,epochKL,epochTotal=0.0,0.0,0.0
    VAE.train()
    
    for batch,x in enumerate(trainSet):
        #reset the grad start each epoch so it does not accumulate
        
        optimiser.zero_grad()

        x=x.to(DEVICE)
   
        xRecon,mu,logvar=VAE(x)

        reconLoss=lossFN(xRecon,x)

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
        sampling(epoch,VAE,trainSet)
    print(f" Training Epoch {epoch}->ReconLoss: {epochRecon} | KLLoss: {epochKL} | TotalLoss: {epochTotal}")
    return epochRecon,epochKL,epochTotal

def validationStep(epoch,BETA,validSet,VAE,lossFN):
    epochValRecon,epochValKL,epochValTotal=0.0,0.0,0.0

    for batch,x in enumerate(validSet):
        #Set to eval to eliminate the batch norm layers 
        VAE.eval()

        with torch.no_grad():
            x=x.to(DEVICE)
       
         
            xRecon,mu,logvar=VAE(x)
        
            reconValLoss=lossFN(xRecon,x)
        
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

def retrievePercentiel(modelFillePath,BETAVAE,validSet,lossThresh,percentile=0.95):

    if modelFillePath is not None:
        loadDict=torch.load(modelFillePath,map_location=DEVICE)
        BETAVAE.load_state_dict(loadDict['modelStateDict'])

    BETAVAE.eval()

    listReconLoss=[]
    with torch.no_grad():
        for batch,x in enumerate(validSet):

            x=x.to(DEVICE)

            xReconed,mu,sigman=BETAVAE(x)

            #get the reconstruction loss for the  batch of images
            reconLoss=lossThresh(xReconed,x)
            #Reduce to be reconstruction loss for one image

            #beacause redeuce is none end up with loss for ever pixel in every chanell etc
            #so flatten it all down to 1 massive tensor dim
            #sum up on that dim  then we end up with the recon loss for eahc iamge in the batch
            perImageLoss=reconLoss.flatten(1).sum(dim=1)

            listReconLoss.append(perImageLoss)


        #take arrays of loss and put them in one lonnnng tensor

        allReconLoss=torch.cat(listReconLoss)

        threshold=torch.quantile(allReconLoss,percentile)
    return threshold




def anomalyCheck(BetaVAE,modelFillePath,dataset,lossThresh,threshhold,accuracy,epoch):

    if modelFillePath is not None:
        loadDict=torch.load(modelFillePath,map_location=DEVICE)
        BetaVAE.load_state_dict(loadDict['modelStateDict'])
   
    BetaVAE.eval()
   

    preds=[]
    true=[]
    for batch,(x,y) in enumerate(dataset):

    
        x=x.to(DEVICE)
   
        xReconed,mu,sigman=BetaVAE(x)
   
        #get the reconstruction loss for the  batch of images
        reconLoss=lossThresh(xReconed,x)
        #Reduce to be reconstruction loss for one image
   
        #beacause redeuce is none end up with loss for ever pixel in every chanell etc
        #so flatten it all down to 1 massive tensor dim
        #sum up on that dim  then we end up with the recon loss for eahc iamge in the batch
        imageLoss=reconLoss.flatten(1).sum(dim=1)

        #mark as anomaly
        #0 disease 
        #1 normal
        if imageLoss>threshhold:
            preds.append(0)

        else:
            preds.append(1)
        true.append(y)




    preds=torch.tensor(preds)
    true=torch.tensor(true)

    #get the metric
    accuracy.update(preds,true)

    acc=accuracy.compute().item()
    recall=recall_score(true,preds,pos_label=0)
    precision=precision_score(true,preds,pos_label=0)



    #clear metrics for next epoch

    accuracy.reset()
    if epoch is not None:
        print(f"Epoch {epoch}->Accuracy: {acc} | Precision: {precision} | Recall: {recall}")  
    else:
        print(f"Peak -> Accuracy:{acc} | Precision:{precision} | Recall:{recall}")
        confMatrix=confusion_matrix(true,preds,labels=[0,1])
        disp=ConfusionMatrixDisplay(confMatrix,display_labels=[0,1])
        disp.plot(cmap='Blues')
        plt.savefig('confMat.png')
        plt.show()
        plt.clf()
 
    return acc,precision,recall

def trainingLoop(BETAVAE,BETA,trainSet,validSet,anomalySet,optimiser,lossFN,lossThresh,modelPath,accuracy):
    #This ensures that beta slowly trends upwards from 0 so we can start with more focus on recon loss and then after 100 epochs we can then focus more on the 
    #KL anealing
    trainReconL=[]
    trainKLL=[]
    trainTotalL=[]
    validReconL=[]
    validKLL=[]
    validTotalL=[]

    accTotal=[]
    precisionTotal=[]
    recallTotal=[]

    bestValidationLoss=float('inf') # absolute biggest loss value possible
    for E in range(NUMEPOCHS):
       ## BETA=sigmoidKlAnnealing(E,0.1,100)
      ##  BETA=min(BETACLAMP,BETA) #clamp it because when beta gets close to 1 alot of reconstruction accuracy is lost
        
    
        trRec,trKL,trTot=trainingStep(E,BETA,trainSet,BETAVAE,optimiser,lossFN)
        vlRec,vlKL,vlTot=validationStep(E,BETA,validSet,BETAVAE,lossFN)
        thresh=retrievePercentiel(None,BETAVAE,validSet,lossThresh,0.95)
        acc,prec,rec=anomalyCheck(BETAVAE,None,anomalySet,lossThresh,thresh,accuracy,E)
        #append the values to array to use in charts later
        trainReconL.append(trRec)
        trainKLL.append(trKL)
        trainTotalL.append(trTot)
        #validation section
        validReconL.append(vlRec)
        validKLL.append(vlKL)
        validTotalL.append(vlTot)

        accTotal.append(acc)
        precisionTotal.append(prec)
        recallTotal.append(rec)

        if vlTot<bestValidationLoss:
            savingModel(E,BETAVAE,vlTot,optimiser,modelPath)
            bestValidationLoss=vlTot

    plotChart('Epoch','Reconstruction Loss','VAE Recon Loss',trainReconL,validReconL,['train','val'],'trainVSvalRecon.png')
    plotChart('Epoch','KL Loss','VAE KL Loss',trainKLL,validKLL,['train','val'],'trainVSvalKLL.png')
    plotChart('Epoch','Total Loss','VAE Total Loss',trainTotalL,validTotalL,['train','val'],'trainVSvalTOT.png')
    plotChart("Epoch",'Accuracy',"Anomaly Detection Accuracy",accTotal,[],['Test'],'Accuracy Test')


def trainVAE():
        #DATASET LOADER
    trainDatasetLoad=RetinalDiseaseLoader(VAETRAIN,True)
    validationDatasetLoad=RetinalDiseaseLoader(VAETEST,False)
    transform=v2.Compose([
        v2.RandomRotation(15),
        v2.ToTensor(),
    ])
    anomalyDatasetLoad=datasets.ImageFolder(VAEANOMALYTEST,transform)

    trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
    validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)
    anomalyset=DataLoader(anomalyDatasetLoad,batch_size=1,shuffle=True)

    anomVAE=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)
    anomVAE=anomVAE.to(DEVICE)


    accuracyMetric=classification.Accuracy(task='binary')
    lossFn=torch.nn.MSELoss(reduction="sum")
    lossThresh=torch.nn.MSELoss(reduction="none")

    optimiser=Adam(anomVAE.parameters(),lr=LR)

    trainingLoop(anomVAE,BETA,trainSet,validSet,anomalyset,optimiser,lossFn,lossThresh,'bestModel.tar',accuracyMetric)

def trainCNN():
            #DATASET LOADER
        transformTrain=v2.Compose([
            v2.RandAugment(3,3),
            v2.ToTensor(),
        ])

        transformValid=v2.Compose([
            v2.ToTensor(),
        ])

        trainDatasetLoad=datasets.ImageFolder(CNNTRAIN,transformTrain)
        validationDatasetLoad=datasets.ImageFolder(CNNTEST,transformValid)
    
    
        trainSet=DataLoader(trainDatasetLoad,batch_size=BATCHSIZE,shuffle=True)
        validSet=DataLoader(validationDatasetLoad,batch_size=BATCHSIZE,shuffle=False)
        

        detectionCNN=AnomalyCNN(inputChannels=INPUTCHANNELS,numCategories=3,rdim=RDIM)
        detectionCNN=detectionCNN.to(DEVICE)
        lossFn=torch.nn.CrossEntropyLoss()
        validLossFn=torch.nn.CrossEntropyLoss()


        optimiser=Adam(detectionCNN.parameters(),lr=LR)
        #Metrics
        accuracyMetric=classification.Accuracy(task='binary').to(DEVICE)
        validAccuracyMetric=classification.Accuracy(task='binary').to(DEVICE)
        precisionMetric=classification.Precision(task='binary').to(DEVICE)
        recallMetric=classification.Recall(task='binary').to(DEVICE)

      

        runTrainingCNN(detectionCNN,trainSet,validSet,lossFn,validLossFn,optimiser,accuracyMetric,validAccuracyMetric,precisionMetric,recallMetric)
        finalTestCNN(detectionCNN,None,validSet,None,None,None)

def loadImage(path):

    transformValid=v2.Compose([
         
            v2.Resize((224,224)),
            v2.ToTensor(),
        ])

        
    return transformValid(Image.open(path).convert('RGB')).unsqueeze(0).to(DEVICE)

def predictCNN(modelPath):
        
    if modelPath is not None:
            loadDict=torch.load(modelPath,map_location=DEVICE)
            detectionCNN=AnomalyCNN(INPUTCHANNELS,3,RDIM)
            detectionCNN.load_state_dict(loadDict['modelStateDict'])
            detectionCNN=detectionCNN.to(DEVICE)
    else:
        return
    root=tk.Tk()
    root.withdraw()
    path=filedialog.askopenfilename(
        title='Select a file',
    )
    tensorIM=loadImage(path)
    detectionCNN.eval()
    with torch.no_grad():
        prediction=torch.argmax(detectionCNN(tensorIM),dim=1)
        indexPred=prediction.item()

        if indexPred==0:
            print("Predicting image as Diseased")
        else:
            print("Predicting image as Healthy")
   
def generate(modelPath):
            
    if modelPath is not None:
            loadDict=torch.load(modelPath,map_location=DEVICE)
            anomaVae=AnomalyVariationalAutoEncoder(INPUTCHANNELS,LATENTDIM,RDIM,LATENTSHAPE)
            anomaVae.load_state_dict(loadDict['modelStateDict'])
            anomaVae=anomaVae.to(DEVICE)
    else:
        return
    anomaVae.eval()
    with torch.no_grad():

        z = torch.randn(1, LATENTDIM)  # sample from gaus normal and see if we can get good looking recons
        z=z.to(DEVICE)

        randomSample=anomaVae.decoder(z)


        randomSample=randomSample.view(-1,3,224,224)
        xRanRec=randomSample.to('cpu')
        save_image(xRanRec,f"generatedImage.png")

        plt.title("Generated image")
        image=img.imread('generatedImage.png')
        plt.imshow(image)
        plt.show()
        
def main():

    MENU = """
    ==== Models ====
    1) Train VAE
    2) Train CNN
    3) Predict with VAE 
    4) Predict with CNN 
    5) Generate image
    0) Quit
    """

    
    while True:
        print(MENU)
        choice = input("Choose: ").strip()
        match choice:
            case "1":
                trainVAE()
            case "2":
                trainCNN()
            case "3":
                predict_vae()
            case "4":
                predictCNN(CNNBEST)
            case "5":
                generate(VAEBEST)                 
            case "0":
                break
            case _:
                print("Invalid choice")
    
main()