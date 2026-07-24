import torch
from torch import nn
from torch.nn import functional as F


class SignLanguageCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 29,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero!")

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.dropout = nn.Dropout(p=0.5)

        self.fully_connected1 = nn.Linear(
            256 * 12 * 12,
            512,
        )

        self.fully_connected2 = nn.Linear(
            512,
            num_classes,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.pool(F.relu(self.conv1(inputs)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))

        output = torch.flatten(
            output,
            start_dim=1,
        )

        output = F.relu(self.fully_connected1(output))
        output = self.dropout(output)

        return self.fully_connected2(output)
