import os
import cv2 as cv
import sys


import copy
import numpy as np
from tqdm import tqdm
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer


import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

sns.set()

device = "cuda:7" if torch.cuda.is_available() else "cpu"


train = os.listdir("/ASR_Kmeans/QUAM/MNIST/x_train_centroids")
test = os.listdir("/ASR_Kmeans/QUAM/MNIST/OOD")


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

x_train = np.empty((len(train), 28, 28))
y_train = np.zeros((len(train), 10))
x_test = np.empty((len(test), 28, 28))


for i in range(len(test)):
    if plt.imread("/ASR_Kmeans/QUAM/MNIST/OOD/" + str(test[i])).shape != (28, 28):
        x_test[i] = np.mean(
            cv.resize(
                cv.imread("/ASR_Kmeans/QUAM/MNIST/OOD/" + str(test[i])), (28, 28)
            ),
            axis=2,
        )
    else:
        x_test[i] = np.mean(
            cv.imread("/ASR_Kmeans/QUAM/MNIST/OOD/" + str(test[i])), axis=2
        )

for i in range(len(train)):
    x_train[i] = np.mean(
        cv.imread("/ASR_Kmeans/QUAM/MNIST/x_train_centroids/" + str(train[i])), axis=2
    )
    y_train[i][dic_y[train[i].split("_")[0]]] = 1


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = np.expand_dims(x_train, 1)

x_train = torch.tensor(x_train).to(device=device, dtype=torch.float32)
y_train = torch.tensor(y_train).to(device=device, dtype=torch.float32)


x_test = np.expand_dims(x_test, 1)

x_test = np.expand_dims(x_test, 1)


x_test = torch.tensor(x_test).to(device=device, dtype=torch.float32)


adversaries = 5
optim_steps = 15
adv_iterations = 10
lr_adv = 0.001
gamma = 0
c_0 = 1e-0
eta = 3
ce = nn.CrossEntropyLoss()


g_cpu = torch.Generator(device="cpu")
n_classes = 10
weight_decay = 1e-3
batch_size = 32


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


network = torch.load("/ASR_Kmeans/QUAM/models/lenet_model.pth")
network.to(device)

with torch.no_grad():
    train_net_pred = network.forward(x_train)
    train_net_ce = ce(train_net_pred, y_train)

p = np.zeros((len(test), 10))

# Adversarial Attack
for a, l in product(range(adversaries), range(n_classes)):

    y_test = torch.LongTensor([l]).to(device=device)

    for i in tqdm(
        range(len(x_test)),
        desc=f"inference {a * n_classes + l + 1} / {adversaries * n_classes}",
    ):

        adversarial_network = copy.deepcopy(network)
        adversarial_network.train()

        # Freeze layers except self.out
        for param in adversarial_network.parameters():
            param.requires_grad = False

        adversarial_network.out.weight.requires_grad = True
        adversarial_network.out.bias.requires_grad = True

        opt = optim.Adam(
            params=adversarial_network.out.parameters(),
            lr=lr_adv,
            weight_decay=weight_decay,
        )
        # opt = optim.Adam(params=adversarial_network.parameters(), lr=lr_adv, weight_decay=weight_decay)
        preds = list()

        c = c_0
        for ad_i in range(adv_iterations):
            for op_s in range(optim_steps):
                idx = torch.randperm(x_train.size(0))[:batch_size]
                train_adv_pred = adversarial_network.forward(x_train[idx])
                penalty = ce(train_adv_pred, y_train[idx])

                test_adv_pred = adversarial_network.forward(x_test[i])
                objective = ce(test_adv_pred, y_test)

                loss = objective + c * (penalty - train_net_ce - gamma)

                with torch.no_grad():
                    adversarial_network.eval()
                    preds.append(
                        torch.softmax(adversarial_network.forward(x_test[i]), dim=1)
                        .cpu()
                        .numpy()
                    )
                    adversarial_network.train()

                opt.zero_grad()
                loss.backward()
                opt.step()

            # update parameter
            c *= eta

        p[i][np.argmax(preds[149][0])] += 1

entropies = []

p = p / 50.0

for i in range(len(p)):
    e = np.sum((-1) * p[i] * np.log(p[i] + 0.0000001))
    entropies.append(e)

print(np.mean(np.array(entropies)))

with open("/ASR_Kmeans/QUAM/entropies_OOD_c.txt", "w+") as f:

    for e in entropies:
        f.write("%s\n" % e)

    print("File written successfully")

f.close()
