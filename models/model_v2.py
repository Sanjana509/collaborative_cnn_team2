
import torch
import torch.nn as nn
from torchvision import models

class ModelV2(nn.Module):
    def __init__(self):
        super(ModelV2, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze feature layers
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # IMPROVEMENT: Dropout + Linear Head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.model(x)
