
import torch
import torch.nn as nn
from torchvision import models

class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        

        for param in self.model.features.parameters():
            param.requires_grad = False
            

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
