
import torch
import torch.nn as nn
from torchvision import models

class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()
        # Load MobileNetV2 with default (pretrained) weights
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze feature layers to make training fast
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Replace the classifier head for 2 classes (Cat vs Dog)
        # MobileNetV2 classifier[1] is the linear layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
