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

noise_vals = np.arange(0, 25, 2)

test_dirs = os.listdir("/ASR_Kmeans/QUAM/MNIST/noise")

test_imgs = os.listdir("/ASR_Kmeans/QUAM/MNIST/x_test")

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

x_test = np.empty((len(test_dirs), len(test_imgs), 28, 28))
y_test = np.zeros((len(test_dirs), len(test_imgs), 10))


for i in range(len(test_dirs)):
    for j in range(len(test_imgs)):
        x_test[i][j] = plt.imread(
            "/ASR_Kmeans/QUAM/MNIST/noise/"
            + str(noise_vals[i])
            + "/"
            + str(test_imgs[j])
        )
        y_test[i][j][dic_y[test_imgs[j].split("_")[0]]] = 1


x_test = np.expand_dims(x_test, 2)
x_test = np.expand_dims(x_test, 2)

x_test = torch.tensor(x_test).float()

p = np.zeros((len(noise_vals), len(test_imgs), 1))

for i in range(len(noise_vals)):
    for j in range(len(test_imgs)):
        with torch.no_grad():
            p[i][j] = np.argmax(network.forward(x_test[i][j]).numpy())


accs = []
for i in range(len(noise_vals)):
    c = 0
    for j in range(len(test_imgs)):
        if p[i][j] == np.argmax(y_test[i][j]):
            c += 1

    accs.append(c / 10.0)

plt.plot(noise_vals, accs)
plt.xlabel("std of gaussian noise")
plt.ylabel("% accuracy")
plt.title("variation of test accuracy with amount of noise")
plt.savefig("/ASR_Kmeans/QUAM/accuracy_plot.png")
