import torch
import torch.nn as nn
import timm
from torch.autograd import Function
import torchvision.models as models
import os
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(vgg19.features))  # Use only the feature extractor part
        self.feature_dim = vgg19.classifier[0].in_features  # The input features to the classifier
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, num_classes=3, num_domains=2):
        super(CNNModel, self).__init__()

        # Feature extractor using the dynamically created VisionEncoder
        self.feature = VisionEncoder()
        self.feature_size = self.feature.feature_dim

        # Improved classification network with residual connections
        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.BatchNorm1d(self.feature_size // 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size // 2, self.feature_size // 4),
            nn.BatchNorm1d(self.feature_size // 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size // 4, num_classes)
        )

        # Improved domain classifier network with residual connections
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.BatchNorm1d(self.feature_size // 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size // 2, self.feature_size // 4),
            nn.BatchNorm1d(self.feature_size // 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size // 4, num_domains)
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output