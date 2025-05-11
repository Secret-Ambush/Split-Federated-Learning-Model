# models.py
import torch.nn as nn

class SplitCNN(nn.Module):
    def __init__(self, in_channels=3):
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

        self.classifier = nn.Linear(128, 10)

    def forward_until(self, x, cut_layer):
        if cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        else:
            raise ValueError("Invalid cut layer")

    def forward_from(self, x, cut_layer):
        if cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        elif cut_layer == 3:
            pass
        else:
            raise ValueError("Invalid cut layer")
        return self.classifier(x)

    def forward(self, x):
        # Optional: Full pass for debugging or test purposes
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
