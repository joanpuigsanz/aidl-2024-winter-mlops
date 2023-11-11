import torch
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator = torch.manual_seed(45)
results_path = "results"
snapshot_path = os.path.join(results_path, "snapshots")
loss_acc = os.path.join(results_path, "loss_acc.txt")


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


# def eval_single_epoch(...):
#     pass


@torch.no_grad()  # decorator: avoid computing gradients
def test_epoch(
    test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion
) -> Tuple[float, float]:
    # Dectivate the train=True flag inside the model
    model.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output, target).item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

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
    return test_loss, test_acc


def train_model(config):
    dataset = MyDataset(
        images_path=os.path.join("data", "chinese_mnist", "data", "data"),
        labels_path=os.path.join("data", "chinese_mnist", "chinese_mnist.csv"),
    )
    train_percent = 0.7
    # eval_percent = 0
    test_percent = 0.3

    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_percent, test_percent], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    # eval_loader = torch.utils.data.DataLoader(
    #     eval_set, batch_size=config["batch_size"], shuffle=False, num_workers=2
    # )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # print("train set")
    # show_images_from_loader(train_set, train_loader)

    # print("test set")
    # show_images_from_loader(test_set, test_loader)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
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
        # eval_single_epoch(...)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # TODO: Compute & save the average test loss & accuracy for the current epoch
        # HELP: Review the functions previously defined to implement the train/test epochs
        test_loss, test_accuracy = test_epoch(
            criterion=criterion, test_loader=test_loader, model=model
        )

        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        with open(loss_acc, "a") as f:
            f.write(f"Epoch {epoch}:\n\ttrain_loss={train_loss:.6f}\ttrain_acc={train_acc:.6f}\n\ttest_loss={test_loss:.6f}\ttest_accuracy={test_accuracy:.6f}\n")
        save_model(model, os.path.join(snapshot_path, f"{epoch}-model.pth"))

    # Plot the plots of the learning curves
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("NLLLoss")
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy [%]")
    plt.plot(train_accs, label="train")
    plt.plot(test_accs, label="test")
    plt.show()
    return model


if __name__ == "__main__":
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(snapshot_path, exist_ok=True)
    try:
        os.remove(loss_acc)
    except OSError as e:
        pass

    config = {
        "batch_size": 64,
        "num_epochs": 10,
        "test_batch_size": 64,
        # "num_classes": 15,
        # "num_inputs": 64 * 64,  # w*h*channels
        "learning_rate": 1e-3,
        "log_interval": 10,
    }
    print(f"Using {device} device")
    train_model(config)
