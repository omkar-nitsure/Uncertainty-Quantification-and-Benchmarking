import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

import copy

map = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

path_train = "/ASR_Kmeans/QUAM/MNIST/x_train"

imgs = os.listdir(path_train)

x_train = []
y_train = []

for i in range(len(imgs)):

    if map[imgs[i].split("_")[0]] != 9:
        x = np.empty((28, 28))
        y = np.zeros(9)
        x = plt.imread(path_train + "/" + imgs[i])
        y[map[imgs[i].split("_")[0]]] = 1
        x_train.append(x)
        y_train.append(y)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.expand_dims(x_train, 1)


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


model = LeNet(9, 0.1, True)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
train = data.TensorDataset(x_train, y_train)
train = data.DataLoader(train, batch_size=32, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 100


def train_model(model, optimizer, train):
    model.train(True)

    # training for given epochs
    for epoch in range(n_epochs):

        epoch_loss = 0

        for _, x in enumerate(train):

            # extracting the input and output from the dataloader
            x_t, y_actual = x[0], x[1]

            # finding the model output
            output = model(x_t)

            # computing the loss for the model output
            loss = loss_fn(output, y_actual)

            # resetting the gradients of the optimizer
            optimizer.zero_grad()

            # computing the gradients of the loss with respect to the model parameters
            loss.backward()

            # updating the model parameters using the optimizer
            optimizer.step()

            epoch_loss += loss.item()

        print("epoch", epoch, " -> ", epoch_loss)


train_model(model, optimizer, train)

torch.save(model, "/ASR_Kmeans/models_MNIST/model.pt")
