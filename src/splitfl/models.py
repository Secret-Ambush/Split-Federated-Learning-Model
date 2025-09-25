# models.py
import torch
import torch.nn as nn
import torchvision.models as models

# ------------------------------
# Base CNN used
# ------------------------------
class SplitCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SplitCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x
        else:
            raise ValueError("Invalid cut layer")

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        elif cut_layer == 3:
            pass
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ------------------------------
# ResNet18
# ------------------------------
class SplitResNet18(nn.Module):
    def __init__(self, num_classes):
        super(SplitResNet18, self).__init__()
        base_model = models.resnet18(weights=None)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.block1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.block2 = nn.Sequential(base_model.layer1, base_model.layer2)
        self.block3 = nn.Sequential(base_model.layer3, base_model.layer4)
        self.classifier = nn.Sequential(base_model.avgpool, nn.Flatten(), base_model.fc)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ------------------------------
# AlexNet
# ------------------------------
class SplitAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(SplitAlexNet, self).__init__()
        base_model = models.alexnet(weights=None)

        # 🚫 Disable final maxpool that collapses small input
        base_model.features[12] = nn.Identity()

        self.block1 = nn.Sequential(*base_model.features[:3])   # Conv1 + ReLU + MaxPool
        self.block2 = nn.Sequential(*base_model.features[3:6])  # Conv2 + ReLU + MaxPool
        self.block3 = nn.Sequential(*base_model.features[6:])   # Remaining convs (no pool)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))       # 🌟 Critical to fix FC input size
        base_model.classifier[6] = nn.Linear(4096, num_classes) # Adjust FC layer
        self.classifier = base_model.classifier

    def forward_until(self, x, cut_layer):
        if cut_layer == 0 or cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            return self.block2(self.block1(x))
        elif cut_layer == 3:
            return self.block3(self.block2(self.block1(x)))
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block3(self.block2(self.block1(x)))
        elif cut_layer in [0, 1]:
            x = self.block3(self.block2(x))
        elif cut_layer == 2:
            x = self.block3(x)
        elif cut_layer == 3:
            pass  # already processed

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ------------------------------
# DenseNet121
# ------------------------------
class SplitDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(SplitDenseNet121, self).__init__()
        base_model = models.densenet121(weights=None)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        self.block1 = base_model.features[:4]
        self.block2 = base_model.features[4:6]
        self.block3 = base_model.features[6:]
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), base_model.classifier)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ------------------------------
# EfficientNet-B0
# ------------------------------
class SplitEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(SplitEfficientNetB0, self).__init__()
        base_model = models.efficientnet_b0(weights=None)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        self.block1 = base_model.features[:2]
        self.block2 = base_model.features[2:5]
        self.block3 = base_model.features[5:]
        self.classifier = nn.Sequential(base_model.avgpool, nn.Flatten(), base_model.classifier)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
