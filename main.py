import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixTransformer(nn.Module):
    def __init__(self):
        super(MatrixTransformer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        print(self.conv1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


model = MatrixTransformer()

input_tensor = torch.randn(1, 1, 28, 28)
output_tensor = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
