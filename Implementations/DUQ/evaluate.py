import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from datasets import get_id, get_ood, get_MNIST_ood, get_omniglot, get_noisy_id


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


def prepare_dataset(ds):

    ds = np.expand_dims(ds, 1)
    ds = torch.tensor(ds).float()

    return ds


def loop_over_dataloader(model, dataloader):
    model.eval()

    with torch.no_grad():
        dists = []
        for data in dataloader:
            data = data.cuda()

            _, dist = model(data)

            dists.append(dist.cpu().numpy())

    dists = np.concatenate(dists)

    return dists


model = torch.load("/ASR_Kmeans/DUQ/models/LeNet_0.5.pt")

dists_list = []
std_vals = []

for i in range(201):
    var = i/200
    if(i % 40 == 0 or i == 200):
        std_vals.append(var)
        test, _ = get_noisy_id(var)
        test = test[np.random.randint(0, len(test), 500)]
        test = prepare_dataset(test)
        test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
        dists_id = loop_over_dataloader(model, test)
        dists_id = np.min(dists_id, axis=1)
        # dists_id = np.mean(dists_id)
        dists_list.append(dists_id)

plt.boxplot(dists_list)


plt.xticks([(i + 1) for i in range(len(dists_list))], [str(std_vals[i]) for i in range(len(std_vals))])
plt.xlabel("std values of gaussian noise added")
plt.ylabel("quantile distribution of the uncertainty values")
plt.title("variation of uncertainty values with noise")

# plt.plot(np.arange(200)/200, dists_list)
# plt.xlabel("std of gaussian noise added")
# plt.ylabel("uncertainty values")
# plt.title("variation of uncertainty values with amount of noise added")
plt.savefig("noise_boxplot.png")

# test, _ = get_id()
# test = test[np.random.randint(0, len(test), 500)]
# ood = get_ood()
# ood = ood[np.random.randint(0, len(ood), 500)]
# ood_MNIST = get_MNIST_ood()
# ood_MNIST = ood_MNIST[np.random.randint(0, len(ood_MNIST), 500)]
# omni = get_omniglot()
# omni = omni[np.random.randint(0, len(omni), 500)]

# test = prepare_dataset(test)
# ood = prepare_dataset(ood)
# ood_MNIST = prepare_dataset(ood_MNIST)
# omni = prepare_dataset(omni)

# test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

# ood = torch.utils.data.DataLoader(ood, batch_size=32, shuffle=True)

# ood_MNIST = torch.utils.data.DataLoader(ood_MNIST, batch_size=32, shuffle=True)

# omni = torch.utils.data.DataLoader(omni, batch_size=32, shuffle=True)

# dists_id = loop_over_dataloader(model, test)
# dists_ood = loop_over_dataloader(model, ood)
# dists_ood_MNIST = loop_over_dataloader(model, ood_MNIST)
# dists_omni = loop_over_dataloader(model, omni)

# dists_id = np.min(dists_id, axis=1)
# dists_ood = np.min(dists_ood, axis=1)
# dists_ood_MNIST = np.min(dists_ood_MNIST, axis=1)
# dists_omni = np.min(dists_omni, axis=1)

# plt.boxplot(dists_ood)
# plt.ylabel("uncertainty scores")
# plt.title("box plot for uncertainty scores of Imagenet")
# plt.savefig("imagenet.png")
# plt.close()

# plt.boxplot(dists_ood_MNIST)
# plt.ylabel("uncertainty scores")
# plt.title("box plot for uncertainty scores of my MNIST")
# plt.savefig("MNIST_OOD.png")
# plt.close()

# plt.boxplot(dists_omni)
# plt.ylabel("uncertainty scores")
# plt.title("box plot for uncertainty scores of Omniglot")
# plt.savefig("Omniglot.png")
# plt.close()

# plt.boxplot(dists_id)
# plt.ylabel("uncertainty scores")
# plt.title("box plot for uncertainty scores of ID examples")
# plt.savefig("MNIST.png")
# plt.close()

# # x = [0, 1, 2]
# # y = [np.mean(dists_id), np.mean(dists_ood_MNIST), np.mean(dists_omni)]
# # x_names = ["MNIST", "MNIST OOD", "Omniglot"]

# # plt.xticks(x, x_names)
# # plt.xlabel("datasets")
# # plt.ylabel("uncertainty scores")
# # plt.title("comparison of uncertainty scores of different datasets")
# # plt.plot(x, y)
# # plt.savefig("dataset_comps_exc_imgnet1.png")

# # print(np.mean(dists_ood), np.mean(dists_id), np.mean(dists_ood_MNIST), np.mean(dists_omni))