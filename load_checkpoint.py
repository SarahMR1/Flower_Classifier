from PIL import Image
from math import floor
import torch
from torch import optim, nn
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import matplotlib.pyplot as plt

def load_checkpoint(filepath):
   
    checkpoint = torch.load(filepath)
    model = models.resnet18(pretrained=True)
    classifier = nn.Sequential( nn.Linear(512, 256),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(256, 102),
                               nn.ReLU(),
                               nn.LogSoftmax(dim=1))
                          
    model.fc=classifier                                
    
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model

model=load_checkpoint('checkpoint.pth')
