r"""This files contains you model defination."""
from sys import dont_write_bytecode
import math
import torch.nn as nn


def make_layers(cfg, batch_norm):
    layers = [nn.Conv2d(3, 32, kernel_size=7, stride=2)]
    in_channels = 32
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, layers_cfg, batch_norm=False):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            *make_layers(layers_cfg, batch_norm)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) # flatten(1D KEL)
        x = self.classifier(x)
        return x


cfg = {
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "vgg19": []
} 

def vgg(model_name): 
    return VGG(cfg[model_name], batch_norm=True)
    