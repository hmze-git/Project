import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
import numpy as np
from PIL import Image


transform=v2.Compose([
    v2.RandomRotation(15),
    v2.ToTensor(),
])

dSet=datasets.ImageFolder(r'C:\Users\hamza\OneDrive\Desktop\HOnours\Advanced Ai\Proj Data\archive\process2\trainAnomaly',transform)


print(f"classes: {dSet.__len__}")