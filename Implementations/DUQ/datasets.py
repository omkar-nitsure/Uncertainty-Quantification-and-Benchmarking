import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms

from scipy.io import loadmat
from PIL import Image

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
    "ood": 10
}


def get_train():

    path1 = "/ASR_Kmeans/QUAM/MNIST/x_train"
    path2 = "/ASR_Kmeans/QUAM/MNIST/OOD"
    imgs1 = os.listdir(path1)
    imgs2 = os.listdir(path2)
    imgs2 = imgs2[:800]
    x_train = np.empty((len(imgs1) + len(imgs2), 28, 28))
    y_train = np.empty(len(imgs1) + len(imgs2), dtype=np.int64)
    id = 0

    for i in range(len(imgs1)):
        x_train[id] = plt.imread(path1 + "/" + imgs1[i])
        y_train[id] = map[imgs1[i].split("_")[0]]
        id += 1


    for i in range(len(imgs2)):
        if(len(plt.imread(path2 + "/" + imgs2[i]).shape) == 3):
            x_train[id] = cv.resize(np.mean(plt.imread(path2 + "/" + imgs2[i]), axis=2), (28, 28))/255.0
        else:
            x_train[id] = cv.resize(plt.imread(path2 + "/" + imgs2[i]), (28, 28))/255.0
        y_train[id] = 10

        id += 1

    return x_train, y_train

def get_id():
    path = "/ASR_Kmeans/QUAM/MNIST/x_test"
    imgs = os.listdir(path)
    x_test = np.empty((len(imgs), 28, 28))
    y_test = np.empty(len(imgs), dtype=np.int64)

    for i in range(len(imgs)):
        x_test[i] = plt.imread(path + "/" + imgs[i])
        y_test[i] = map[imgs[i].split("_")[0]]

    return x_test, y_test


def get_test():
    path1 = "/ASR_Kmeans/QUAM/MNIST/x_test"
    imgs1 = os.listdir(path1)
    path2 = "/ASR_Kmeans/QUAM/MNIST/OOD"
    imgs2 = os.listdir(path2)
    imgs2 = imgs2[800:]
    x_test = np.empty((len(imgs1) + len(imgs2), 28, 28))
    y_test = np.empty(len(imgs1) + len(imgs2), dtype=np.int64)
    id = 0

    for i in range(len(imgs1)):
        x_test[id] = plt.imread(path1 + "/" + imgs1[i])
        y_test[id] = map[imgs1[i].split("_")[0]]
        id += 1

    for i in range(len(imgs2)):
        if(len(plt.imread(path2 + "/" + imgs2[i]).shape) == 3):
            x_test[id] = cv.resize(np.mean(plt.imread(path2 + "/" + imgs2[i]), axis=2), (28, 28))/255.0
        else:
            x_test[id] = cv.resize(plt.imread(path2 + "/" + imgs2[i]), (28, 28))/255.0
        y_test[id] = 10
        id += 1

    return x_test, y_test


def get_ood():
    path = "/ASR_Kmeans/QUAM/MNIST/OOD"
    imgs = os.listdir(path)
    x_ood = np.empty((len(imgs), 28, 28))

    for i in range(len(imgs)):
        if len(plt.imread(path + "/" + imgs[i]).shape) == 3:
            x_ood[i] = np.array(
                cv.resize(np.mean(plt.imread(path + "/" + imgs[i]), axis=2), (28, 28))
                / 255.0
            )
        else:
            x_ood[i] = plt.imread(path + "/" + imgs[i])

    return x_ood

def get_MNIST_ood():
    path = "/ASR_Kmeans/QUAM/MNIST/OOD_MNIST"
    imgs = os.listdir(path)
    x_ood = np.empty((len(imgs), 28, 28))

    for i in range(len(imgs)):
        if len(plt.imread(path + "/" + imgs[i]).shape) == 3:
            x_ood[i] = np.array(
                cv.resize(np.mean(plt.imread(path + "/" + imgs[i]), axis=2), (28, 28))
                / 255.0
            )
        else:
            x_ood[i] = cv.resize(plt.imread(path + "/" + imgs[i]), (28, 28))/255.0

    return x_ood

def get_omniglot():
    
    path = "/ASR_Kmeans/QUAM/MNIST/OOD_MNIST"
    imgs = os.listdir(path)
    x_ood = np.empty((len(imgs), 28, 28))

    for i in range(len(imgs)):
        if len(plt.imread(path + "/" + imgs[i]).shape) == 3:
            x_ood[i] = np.array(
                cv.resize(np.mean(plt.imread(path + "/" + imgs[i]), axis=2), (28, 28))
                / 255.0
            )
        else:
            x_ood[i] = cv.resize(plt.imread(path + "/" + imgs[i]), (28, 28))/255.0

    return x_ood


def get_noisy_id(var):
    path = "/ASR_Kmeans/QUAM/MNIST/x_test"
    imgs = os.listdir(path)
    x_test = np.empty((len(imgs), 28, 28))
    y_test = np.empty(len(imgs), dtype=np.int64)

    for i in range(len(imgs)):
        x_test[i] =  + np.random.normal(plt.imread(path + "/" + imgs[i]), var*np.max(plt.imread(path + "/" + imgs[i])))
        y_test[i] = map[imgs[i].split("_")[0]]

    return x_test, y_test