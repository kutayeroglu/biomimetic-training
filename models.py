import torch
import torch.nn as nn
import torch.optim as optim


# 1. The Model Architecture (Replicating 'alexnet22_48')
class AlexNetModified(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetModified, self).__init__()

        # Paper & models.py: Modified first layer (48 filters, 22x22 kernel)
        # Padding calculation: 'same' in TF for stride 1 means padding = (kernel_size - 1) / 2

        self.features = nn.Sequential(
            # Layer 1: Conv2D(48, kernel=(22,22), strides=4, padding='valid')
            nn.Conv2d(3, 48, kernel_size=22, stride=4, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 2: Conv2D(256, kernel=(5,5), strides=1, padding='same')
            nn.Conv2d(48, 256, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 3: Conv2D(384, kernel=(3,3), strides=1, padding='same')
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            # Layer 4: Conv2D(384, kernel=(3,3), strides=1, padding='same')
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            # Layer 5: Conv2D(256, kernel=(3,3), strides=1, padding='same')
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dense(4096)
            # Input size depends on the output of features. For 227x227 input:
            # L1: (227-22)/4 + 1 = 52 -> Pool -> 25
            # L2: 25 -> Pool -> 12
            # L5: 12 -> Pool -> 5
            # Final feature map is 256 * 5 * 5 = 6400 (Check standard AlexNet sizing usually 6x6, but 22 kernel changes this)
            # Let's use a dummy pass to determine linear input size or assume standard implementation logic.
            # *Adjustment*: With 22x22 kernel/stride 4 on 227 input, output is roughly 5x5 spatial.
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 2. Data Transforms (Replicating 'Rescaling' and 'RandomCrop')
def get_transforms():
    # Paper: Rescaling [-1, 1], Random Horizontal Flip, Random Crop to 227x227 [cite: 277]
    return transforms.Compose(
        [
            transforms.Resize(
                256
            ),  # Resize usually precedes cropping in ImageNet pipelines
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converts [0, 255] to [0.0, 1.0]
            # Normalize to get [-1, 1]: (x - 0.5) / 0.5 = 2x - 1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# 3. Training Configuration (Standard Regimen)
def get_training_setup(model):
    # Paper: Optimizer SGD, Momentum 0.9, LR 0.001
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    # Paper: Categorical Cross Entropy
    criterion = nn.CrossEntropyLoss()

    return optimizer, criterion


# Usage Example
if __name__ == "__main__":
    # Initialize Model
    model = AlexNetModified(num_classes=1000)

    # Dummy pass: verify layer dimensions (imitate cropped input dims)
    dummy_input = torch.randn(1, 3, 227, 227)
    out = model.features(dummy_input)
    print(
        f"Feature output shape: {out.shape}"
    )  # Use this to fix the linear layer input size

    # Setup
    optimizer, criterion = get_training_setup(model)

    print("Standard Regimen (AlexNet Modified) Ready for Training.")
