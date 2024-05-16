import torch
import torch.nn as nn
from torch.nn import functional as F
import copy


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


class SoftmaxModel(LeNet):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.last_layer = nn.Linear(84, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.last_layer(self.project(x))
        y_pred = F.log_softmax(z, dim=1)

        return y_pred
