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

test = os.listdir("/ASR_Kmeans/QUAM/MNIST/x_train")

dic_y = {
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

x_test = np.empty((len(test), 28, 28))
y_test = np.zeros((len(x_test), 10))

for i in range(len(test)):
    x_test[i] = plt.imread("/ASR_Kmeans/QUAM/MNIST/x_train/" + str(test[i]))
    y_test[i][dic_y[test[i].split("_")[0]]] = 1

# x_test = x_test/255.0

x_test = np.expand_dims(x_test, 1)
x_test = np.expand_dims(x_test, 1)


x_test = torch.tensor(x_test).float()

p = []

for i in range(len(x_test)):
    with torch.no_grad():
        a = np.argmax(network.forward(x_test[i]).numpy())
        p.append(a)


c = 0
nc = []
for i in range(len(p)):
    if p[i] == np.argmax(y_test[i]):
        c += 1
    else:
        nc.append(i)


with open("/ASR_Kmeans/QUAM/incorrect.txt", "w+") as f:

    for e in nc:
        f.write("%s\n" % e)

print(c)
