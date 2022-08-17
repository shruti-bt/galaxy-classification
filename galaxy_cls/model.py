# add your model hear

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet100":
        pass
    return model 

