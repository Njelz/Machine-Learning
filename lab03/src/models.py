import torch
from torch import nn
from torchinfo import summary


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Linear(7 * 7, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Split each 28x28 image into sixteen 7x7 patches.
        patches = nn.functional.unfold(
            x, kernel_size=7, stride=7
        ).transpose(1, 2)

        tokens = self.patch_projection(patches)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def main():
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    example_batch = torch.zeros(2, 1, 28, 28)

    for name, model in models.items():
        output = model(example_batch)
        model_summary = summary(
            model,
            input_size=(2, 1, 28, 28),
            verbose=0,
        )
        print(
            f"{name}: output shape={tuple(output.shape)}, "
            f"parameters={model_summary.total_params:,}"
        )


if __name__ == "__main__":
    main()
