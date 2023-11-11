import torch
import os
from dataset import MyDataset
import torch.nn.functional as F
from torch.utils.data import Dataset
from model import MyModel
from utils import accuracy, save_model
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

generator = torch.manual_seed(45)


def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> int:
    """
    Define the Accuracy metric in the function below by:
    (1) obtain the maximum for each predicted element in the batch to get the
        class (it is the maximum index of the num_classes array per batch sample)
        (look at torch.argmax in the PyTorch documentation)
    (2) compare the predicted class index with the index in its corresponding
        neighbor within label_batch
    (3) sum up the number of affirmative comparisons and return the summation

    Parameters:
    -----------
    predicted_batch: torch.Tensor shape: [BATCH_SIZE, N_CLASSES]
        Batch of predictions
    label_batch: torch.Tensor shape: [BATCH_SIZE, 1]
        Batch of labels / ground truths.
    """
    pred = predicted_batch.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def train_single_epoch(
    config,
    epoch,
    train_loader: torch.utils.data.DataLoader,
    network: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.functional,
) -> Tuple[float, float]:
    # Activate the train=True flag inside the model
    network.train()

    avg_loss = None
    acc = 0.0
    train_loss = []
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % config["log_interval"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    avg_acc = 100.0 * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc


# def eval_single_epoch(...):
#     pass


@torch.no_grad()  # decorator: avoid computing gradients
def test_epoch(
    criterion,
    test_loader: torch.utils.data.DataLoader,
    network: torch.nn.Module,
) -> Tuple[float, float]:
    # Dectivate the train=True flag inside the model
    network.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = network(data)

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
        images_path=os.path.join("data_set", "data", "data"),
        labels_path=os.path.join("data_set", "chinese_mnist.csv"),
    )
    my_model = MyModel().to(device)

    train_percent = 0.7
    eval_percent = 0.2
    test_percent = 0.1

    train_set, eval_set, test_set = torch.utils.data.random_split(
        dataset, [train_percent, eval_percent, test_percent], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    criterion = F.nll_loss
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=config["learning_rate"])

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_single_epoch(
            config=config,
            epoch=epoch,
            criterion=criterion,
            optimizer=optimizer,
            network=my_model,
            train_loader=train_loader,
        )
        # eval_single_epoch(...)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"accuracy: {train_accs[-1]}")

        # TODO: Compute & save the average test loss & accuracy for the current epoch
        # HELP: Review the functions previously defined to implement the train/test epochs
        test_loss, test_accuracy = test_epoch(
            criterion=criterion, test_loader=test_loader, network=my_model
        )

        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

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
    return my_model


if __name__ == "__main__":
    config = {
        "batch_size": 64,
        "num_epochs": 20,
        "test_batch_size": 64,
        "hidden_size": 128,
        "num_classes": 15,
        "num_inputs": 64 * 64,  # w*h*channels
        "learning_rate": 1e-4,
        "log_interval": 10,
    }
    model = train_model(config)
    save_model(model, "model")
