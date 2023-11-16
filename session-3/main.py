import os
import zipfile

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import MyModel
from utils import binary_accuracy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
        epoch,
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim,
        loss_fn: torch.nn.functional,
        log_interval: int,
):
    model.train()
    accs, losses = [], []
    total_data_analysed = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # You will need to do y = y.unsqueeze(1).float() to add an output dimension to the labels and cast to the correct type
        y = y.unsqueeze(1).float()
        data, target = x.to(device), y.to(device)

        # Set model gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the model
        output = model(data)

        # Compute loss
        loss = loss_fn(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc = binary_accuracy(target, output)

        losses.append(loss.item())
        accs.append(acc.item())
        total_data_analysed += len(data)

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{total_data_analysed}/{len(train_loader.dataset)} ({(100.0 * batch_idx / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}")
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model: torch.nn.Module,
                      val_loader: torch.utils.data.DataLoader,
                      criterion: torch.nn.functional):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            y = y.unsqueeze(1).float()

            data, target = x.to(x), y.to(y)
            output = model(data)

            loss = criterion(output, target)
            acc = binary_accuracy(target, output)

            losses.append(loss.item())
            accs.append(acc.item())
    # Average accuracy across all correct predictions batches now
    val_acc = np.mean(acc)
    val_loss = np.mean(losses)
    print(f"\nTest set: Average loss: {val_loss:.4f}, Accuracy: {(val_acc * 100):.2f}%\n")
    return np.mean(losses), np.mean(accs)


def train_model(config):

    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_dataset = ImageFolder(root=os.path.join("data", "cars_vs_flowers", "dataset", "cars_vs_flowers", "training_set"), transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = ImageFolder(root=os.path.join("data", "cars_vs_flowers", "dataset", "cars_vs_flowers", "test_set"), transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel().to(device)
    # loss_fn = nn.NLLLoss(reduction="mean")
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(my_model.parameters(), config["lr"])

    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(epoch=epoch,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       model=my_model,
                                       train_loader=train_loader,
                                       log_interval=1)
        print(f"Train Epoch: {epoch}  Loss={loss:.6f} Acc={(acc*100):.2f}%")
        loss, acc = eval_single_epoch(model=my_model,
                                      val_loader=test_loader,
                                      criterion=loss_fn)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    return my_model


def unzip_dataset():
    extraction_path = os.path.join("data", "cars_vs_flowers")
    if not os.path.exists(extraction_path):
        with zipfile.ZipFile(os.path.join("data", "cars_vs_flowers.zip"), 'r') as zip_ref:
            zip_ref.extractall(extraction_path)


if __name__ == "__main__":

    unzip_dataset()
    results_path = "results"
    snapshot_path = os.path.join(results_path, "snapshots")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(snapshot_path, exist_ok=True)

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
    }
    my_model = train_model(config)
    torch.save(my_model.state_dict(), os.path.join(snapshot_path, "model.pth"))

    
