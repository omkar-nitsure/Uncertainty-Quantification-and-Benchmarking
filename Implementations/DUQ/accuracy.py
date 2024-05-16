import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import functional as F
import matplotlib.pyplot as plt
import copy

from datasets import get_id


class LeNet(nn.Module):

    def __init__(self, out_features: int, p_drop: float = 0.2, mnist=True):
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


class CNN_DUQ(LeNet):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    ):
        super().__init__(out_features=84, mnist=True)

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 84), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def update_embeddings(self, x, y):
        z = self.last_layer(self.project(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (diff**2).mean(1).div(2 * self.sigma**2)
        exp_dists = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()

        return distances, exp_dists

    def forward(self, x):
        z = self.last_layer(self.project(x))
        dists, y_pred = self.output_layer(z)

        return y_pred, dists


def prepare_datasets(x_test, y_test):

    x_test = np.expand_dims(x_test, 1)

    x_test = torch.tensor(x_test).float()

    y_test = torch.tensor(y_test)

    train = data.TensorDataset(x_test, y_test)

    return data.DataLoader(train, batch_size=1, shuffle=False)


def accuracy(model, dataloader):

    with torch.no_grad():
        predictions = []
        dists = []
        c = 0
        t = 0
        for x, y in dataloader:
            x = x.cuda()
            y = y.cpu().numpy()

            pred, dist = model(x)

            pred = pred.cpu().numpy()
            dist = dist.cpu().numpy()

            pred = np.argmax(pred, axis=1)

            predictions.append(pred)

            dists.append(dist)

            for i in range(len(y)):
                if(y[i] == pred[i]):
                    c += 1
                t += 1

    return  c*100/t

def misclassified(model, dataloader):

    with torch.no_grad():

        misclass = []
        dists = []
        idx = 0
        for x, y in dataloader:
            x = x.cuda()
            y = y.cpu().numpy()

            pred, dist = model(x)

            pred = pred.cpu().numpy()
            dist = dist.cpu().numpy()

            pred = np.argmax(pred, axis=1)


            dists.append(np.min(dist))

            if(y != pred):
                misclass.append(idx)
            idx += 1

    return  np.array(misclass), np.array(dists)



model = torch.load("/ASR_Kmeans/DUQ/models/LeNet_0.3.pt")

x_test, y_test = get_id()

dl = prepare_datasets(x_test, y_test)

acc = accuracy(model, dl)

misclass, dists = misclassified(model, dl)

d, id = zip(*sorted(zip(-dists, np.arange(1000))))

# ans = []

# for p in range(74):
#     arr = id[:10*p]
#     acc = 0
#     for i in range(len(arr)):
#         found = 0
#         for j in range(len(misclass)):
#             if(arr[i] == misclass[j]):
#                 found = 1
#         if(found):
#             acc += 1

#     a = (791 - (10*p - acc))*100/(1000 - 10*p)

#     ans.append(a)


ans = []

for p in range(64):
    arr = id[:10*p]
    acc = 0
    for i in range(len(arr)):
        found = 0
        for j in range(len(misclass)):
            if(arr[i] == misclass[j]):
                found = 1
        if(found):
            acc += 1

    a = (835 - (10*p - acc))*100/(1000 - 10*p)

    ans.append(a)

# ans = np.array(ans)

# print(ans)

plt.plot(np.arange(64), ans)
plt.xlabel("% of highly uncertain points excluded")
plt.ylabel("accuracy")
plt.title("accuracy when uncertain points excluded from inference")
plt.savefig("idea_plot.png")