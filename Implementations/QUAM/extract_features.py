import numpy as np
import torch
import torch.nn as nn
import copy
import os
import matplotlib.pyplot as plt


class LeNet(nn.Module):

    def __init__(self, out_features: int, p_drop: float = 0.2, mnist=False):
        super(LeNet, self).__init__()

        self.first_layer_fm = None
        self.second_layer_fm = None

        activation = nn.ReLU()

        dim = 4 if mnist else 5

        self.first_layer = nn.Conv2d(
            in_channels=1 if mnist else 3, out_channels=6, kernel_size=5
        )

        self.second_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        )

        self.projection = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Flatten(),
            nn.Linear(in_features=16 * dim**2, out_features=120),
            activation,
            nn.Linear(in_features=120, out_features=84),
            activation,
            nn.Dropout(p=p_drop),
        )

        self.out = nn.Linear(in_features=84, out_features=out_features)

    def project(self, x, store_fm=False):
        x = self.first_layer(x)
        if store_fm:
            self.first_layer_fm = copy.deepcopy(x.detach())
        x = self.second_layer(x)
        if store_fm:
            self.second_layer_fm = copy.deepcopy(x.detach())
        return self.projection(x)

    def forward(self, x, store_fm=False):
        return self.out(self.project(x, store_fm))


network = torch.load("/ASR_Kmeans/QUAM/lenet_model.pth")

imgs = os.listdir("/ASR_Kmeans/QUAM/MNIST/x_train")

features = np.empty((len(imgs), 84))

x_train = np.empty((len(imgs), 28, 28))

for i in range(len(imgs)):

    x_train[i] = plt.imread("/ASR_Kmeans/QUAM/MNIST/x_train" + "/" + imgs[i])

x_train = np.expand_dims(x_train, 1)
x_train = np.expand_dims(x_train, 1)

x_train = torch.tensor(x_train).float()

for i in range(len(x_train)):
    with torch.no_grad():
        features[i] = network.project(x_train[i]).numpy()

np.save("features.npy", features)
