from torch import nn
from torchvision import models
class SmallCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
          nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
          nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128,num_classes))
    def forward(self,x): return self.classifier(self.features(x))
def build_classifier(name="cnn",num_classes=6,pretrained=False):
    if name=="cnn": return SmallCNN(num_classes)
    if name=="resnet18":
        m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc=nn.Linear(m.fc.in_features,num_classes); return m
    raise ValueError(f"Unknown classifier: {name}")
