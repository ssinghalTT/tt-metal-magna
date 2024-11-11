# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph


class Conv_mnist_like_model(nn.Module):
    def __init__(self, num_classes):
        super(Conv_mnist_like_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(3 * 3 * 64, 1024)  # 100 x 100 region
        self.fc2 = nn.Linear(1024, num_classes)
        self.Dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool3(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool4(F.relu(self.conv3(x)))
        x = self.pool5(F.relu(self.conv4(x)))
        x = x.view(-1, 3 * 3 * 64)
        x = self.Dropout(x)
        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = self.fc2(x)
        return x


# torch_model = Conv_mnist_like_model(1000)
# input_tensor = torch.randn(1, 5, 94, 94)
# # output = model(input_tensor)
# model_graph = draw_graph(
#     torch_model,
#     input_size=(1, 5, 94, 94),
#     expand_nested=True,  # To expand nested modules in the graph
#     graph_name="conv_mnist"
# )

# # Render the graph and save it as a PDF
# model_graph.visual_graph.render(format="pdf")
