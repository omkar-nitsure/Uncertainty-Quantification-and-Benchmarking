import random
import numpy as np

import torch
import torch.utils.data as data
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from datasets import get_train, get_test
from model import CNN_DUQ


def train_model(l_gradient_penalty, length_scale, final_model):
    x_train, y_train = get_train()
    x_test, y_test = get_test()

    x_val, y_val = x_test[:600], y_test[:600]
    x_test, y_test = x_test[600:], y_test[600:]

    x_train = np.expand_dims(x_train, 1)
    x_val = np.expand_dims(x_val, 1)
    x_test = np.expand_dims(x_test, 1)

    num_classes = 10
    embedding_size = 84
    learnable_length_scale = False
    gamma = 0.999

    model = CNN_DUQ(
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )

    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch

        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred, _ = model(x)

        loss = F.binary_cross_entropy(y_pred, y)
        loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred.sum(1))

        x.requires_grad_(False)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch

        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred, _ = model(x)

        return y_pred, y, x, y_pred.sum(1)

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.2
    )

    train = data.TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train))
    val = data.TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val))
    test = data.TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test))

    dl_train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

    dl_val = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)

    dl_test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

        if trainer.state.epoch % 5 == 0:
            evaluator.run(dl_val)

            metrics = evaluator.state.metrics

            print(
                f"Validation Results - Epoch: {trainer.state.epoch} "
                f"Acc: {metrics['accuracy']:.4f} "
                f"BCE: {metrics['bce']:.2f} "
                f"GP: {metrics['gradient_penalty']:.6f} "
            )
            print(f"Sigma: {model.sigma}")

    trainer.run(dl_train, max_epochs=50)

    evaluator.run(dl_val)
    val_accuracy = evaluator.state.metrics["accuracy"]

    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]

    return model, val_accuracy, test_accuracy


if __name__ == "__main__":

    # Finding length scale - decided based on validation accuracy
    l_gradient_penalties = [0.0]
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    repetition = 1  # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            val_accuracies = []
            test_accuracies = []
            roc_aucs_mnist = []
            roc_aucs_notmnist = []

            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                model, val_accuracy, test_accuracy = train_model(
                    l_gradient_penalty, length_scale, final_model
                )

                torch.save(model, f"/ASR_Kmeans/DUQ/models/LeNet_{length_scale}.pt")
