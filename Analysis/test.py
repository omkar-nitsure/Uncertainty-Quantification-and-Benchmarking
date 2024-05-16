import numpy as np
import matplotlib.pyplot as plt
import os
import copy

import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.manifold import TSNE

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


model = torch.load("/ASR_Kmeans/models_MNIST/model.pt")

path = "/ASR_Kmeans/QUAM/MNIST/x_test"

imgs = os.listdir(path)

x_test = np.empty((len(imgs), 28, 28))

id = 0

for idx in range(10):
    for i in range(len(imgs)):
        if map[imgs[i].split("_")[0]] == idx:

            x_test[id] = plt.imread(path + "/" + imgs[i])
            id += 1

x_test = np.expand_dims(x_test, 1)

x_test = torch.tensor(x_test).float()

test = data.DataLoader(x_test, batch_size=32, shuffle=False)

feats = []

for x in test:
    with torch.no_grad():
        f = model.project(x)
        feats.append(f)

for i in range(len(feats)):
    feats[i] = np.array(feats[i])

f_final = feats[0]
for i in range(1, len(feats)):
    f_final = np.concatenate((f_final, feats[i]))

f_final = np.array(f_final)

tsne = TSNE(n_components=2, n_iter=1000)
tsne_result = tsne.fit_transform(f_final)

zero = tsne_result[0:98]
one = tsne_result[98:219]
two = tsne_result[219:323]
three = tsne_result[323:422]
four = tsne_result[422:529]
five = tsne_result[529:615]
six = tsne_result[615:705]
seven = tsne_result[705:813]
eight = tsne_result[813:895]

# Plot the results
plt.scatter(zero[:, 0], zero[:, 1], label="0", alpha=0.7)
plt.scatter(one[:, 0], one[:, 1], label="1", alpha=0.7)
plt.scatter(two[:, 0], two[:, 1], label="2", alpha=0.7)
plt.scatter(three[:, 0], three[:, 1], label="3", alpha=0.7)
plt.scatter(four[:, 0], four[:, 1], label="4", alpha=0.7)
plt.scatter(five[:, 0], five[:, 1], label="5", alpha=0.7)
plt.scatter(six[:, 0], six[:, 1], label="6", alpha=0.7)
plt.scatter(seven[:, 0], seven[:, 1], label="7", alpha=0.7)
plt.scatter(eight[:, 0], eight[:, 1], label="8", alpha=0.7)


plt.title("T-SNE plot of the MNIST features")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("feature_clusters.png")
