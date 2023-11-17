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
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch import nn
from torch.utils.data import Dataset

# import ray.train
# from ray.train import ScalingConfig
# from ray.train.torch import TorchTrainer
# from ray.tune.schedulers import ASHAScheduler



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator = torch.manual_seed(45)
results_path = "results"
loss_acc = os.path.join(results_path, "loss_acc_optimized.txt")
best_model_path = os.path.join(results_path, "best-model_optimized.pt")


def train_single_epoch(
        epoch,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim,
        loss_fn: torch.nn.functional,
        log_interval: int
):

    model.train()

    accs, losses = [], []
    total_data_analysed = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        # Move input data and labels to the device
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(data)
        # Compute loss
        loss = loss_fn(output, labels)
        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc = accuracy(labels, output)
        accs.append(acc.item())
        losses.append(loss.item())
        total_data_analysed += len(data)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    total_data_analysed,
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    loss.item(),
                    )
            )

    return np.mean(losses), np.mean(accs)


def eval_single_epoch(
        test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, fn_loss
):
    # Dectivate the train=True flag inside the model
    model.eval()

    test_loss = []
    acc = 0

    correct_predictions = 0
    all_targets = []
    all_predictions = []
    conf_matrix = np.zeros((15,15))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # Apply the loss criterion and accumulate the loss
            test_loss.append(fn_loss(output, target).item())

            # compute number of correct predictions in the batch
            acc += compute_accuracy(output, target)

            # Build the confusion matrix
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for i in range(target.size(0)):
                label = target.data[i]
                # Update confusion matrix
                conf_matrix[label][predicted.data[i]] += 1

    # Average accuracy across all correct predictions batches now
    test_acc = 100.0 * acc / len(test_loader.dataset)
    test_loss = np.mean(test_loss)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            acc,
            len(test_loader.dataset),
            test_acc,
        )
    )
    return test_loss, test_acc, conf_matrix


def train_model(config: dict):
    dataset = MyDataset(
        images_path=os.path.join("data", "chinese_mnist", "data", "data"),
        labels_path=os.path.join("data", "chinese_mnist", "chinese_mnist.csv"),
    )
    train_size, val_size, test_size = 10000, 2500, 2500

    train_set, eval_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    # print("train set")
    # show_images_from_loader(train_set, train_loader)

    # print("test set")
    # show_images_from_loader(test_set, test_loader)

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    # model = PseudoLeNet()
    model = MyModel()
    model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    # loss_fn = nn.NLLLoss(reduction="mean")
    loss_fn = nn.CrossEntropyLoss()

    best_eval_acc = 0

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_single_epoch(
            epoch=epoch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            model=model,
            dataloader=train_loader,
            log_interval=config["log_interval"],
        )
        eval_loss, eval_accuracy, conf_matrix = eval_single_epoch(fn_loss=loss_fn, test_loader=eval_loader, model=model)

        eval_losses.append(eval_loss)
        eval_accs.append(eval_accuracy)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    best_model = torch.jit.load(best_model_path)

    test_loss, test_acc, conf_matrix = eval_single_epoch(fn_loss=loss_fn, test_loader=test_loader, model=best_model)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    return { "val_loss": eval_losses[-1]}


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
