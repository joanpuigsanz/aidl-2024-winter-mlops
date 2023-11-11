import torch
import ray
from ray import tune
import os
from dataset import MyDataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from model import MyModel
from model_2 import PseudoLeNet
from utils import accuracy, save_model, compute_accuracy
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator = torch.manual_seed(45)
# results_path = "results"
# snapshot_path = os.path.join(results_path, "snapshots")
# loss_acc = os.path.join(results_path, "loss_acc.txt")


def train_single_epoch(
        epoch,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        log_interval: int,
) -> Tuple[float, float]:
    model.train()

    train_loss = []
    acc = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set model gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the model
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    loss.item(),
                    )
            )
    avg_acc = 100.0 * acc / len(dataloader.dataset)

    return np.mean(train_loss), avg_acc


@torch.no_grad()  # decorator: avoid computing gradients
def eval_single_epoch(eval_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion) -> Tuple[float, float]:
    # Dectivate the train=True flag inside the model
    model.eval()

    eval_loss = []
    acc = 0
    for data, target in eval_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        # Apply the loss criterion and accumulate the loss
        eval_loss.append(criterion(output, target).item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    # Average accuracy across all correct predictions batches now
    eval_acc = 100.0 * acc / len(eval_loader.dataset)
    eval_loss = np.mean(eval_loss)
    print(
        "\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            eval_loss,
            acc,
            len(eval_loader.dataset),
            eval_acc,
        )
    )
    return eval_loss, eval_acc


def train_model(config: dict):
    dataset = MyDataset(
        images_path=os.path.join("data", "chinese_mnist", "data", "data"),
        labels_path=os.path.join("data", "chinese_mnist", "chinese_mnist.csv"),
    )
    train_percent = 0.5
    eval_percent = 0.25
    test_percent = 0.25

    train_set, eval_set, test_set = torch.utils.data.random_split(dataset, [train_percent, eval_percent, test_percent], generator=generator)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    # model = PseudoLeNet()
    model = MyModel()
    model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(reduction="mean")

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_single_epoch(
            epoch=epoch,
            criterion=criterion,
            optimizer=optimizer,
            model=model,
            dataloader=train_loader,
            log_interval=config["log_interval"],
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        eval_loss, eval_accuracy = eval_single_epoch(
            criterion=criterion, eval_loader=test_loader, model=model
        )

        eval_losses.append(eval_loss)
        eval_accs.append(eval_accuracy)

    return eval_losses[-1]


if __name__ == "__main__":
    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            # "hyperparam_1": tune.uniform(1, 10),
            # "hyperparam_2": tune.grid_search(["relu", "tanh"]),
            "batch_size": tune.choice([16, 32, 64]),
            "num_epochs": 10,
            "test_batch_size": 64,
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "log_interval": 10,
        },
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    # print(test_model(...))
