import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset import MyDataset
from model import MyModel
from utils import accuracy, compute_accuracy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator = torch.manual_seed(45)
results_path = "results"
loss_acc = os.path.join(results_path, "loss_acc.txt")
best_model_path = os.path.join(results_path, "best-model.pt")


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


def train_model(config):
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

        if eval_accuracy > best_eval_acc:
            best_eval_acc = eval_accuracy
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save(best_model_path)

        with open(loss_acc, "a") as f:
            f.write(f"Epoch {epoch}:\n\ttrain_loss={train_loss:.6f}\ttrain_acc={train_acc:.6f}\n\ttest_loss={eval_loss:.6f}\ttest_accuracy={eval_accuracy:.6f}\n")

    best_model = torch.jit.load(best_model_path)

    test_loss, test_acc, conf_matrix = eval_single_epoch(fn_loss=loss_fn, test_loader=test_loader, model=best_model)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Plot the plots of the learning curves
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("NLLLoss")
    plt.plot(train_losses, label="train")
    plt.plot(eval_losses, label="test")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Eval Accuracy [%]")
    plt.plot(train_accs, label="train")
    plt.plot(eval_accs, label="test")
    plt.show()

    # Confusion matrix
    # more pretty plots using https://github.com/wcipriano/pretty-print-confusion-matrix
    df_cm = pd.DataFrame(conf_matrix, range(15), range(15))
    plt.figure(figsize=(20, 14))
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(df_cm, fmt='.10g', annot=True, annot_kws={"size": 14}) # font size
    plt.show()

    plt.show()
    return model


def unzip_dataset():
    extraction_path = os.path.join("data", "chinese_mnist")
    if not os.path.exists(extraction_path):
        with zipfile.ZipFile(os.path.join("data", "chinese_mnist.zip"), 'r') as zip_ref:
            zip_ref.extractall(extraction_path)


if __name__ == "__main__":
    # config = {
    #     "batch_size": 64,
    #     "num_epochs": 10,
    #     "test_batch_size": 64,
    #     # "num_classes": 15,
    #     # "num_inputs": 64 * 64,  # w*h*channels
    #     "learning_rate": 1e-3,
    #     "log_interval": 10,
    # }
    # dataset = MyDataset(
    #     images_path=os.path.join("data", "chinese_mnist", "data", "data"),
    #     labels_path=os.path.join("data", "chinese_mnist", "chinese_mnist.csv"),
    # )
    # train_size, val_size, test_size = 10000, 2500, 2500
    #
    # train_set, eval_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
    #
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    # eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config["batch_size"], shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
    #
    # # optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    # # loss_fn = nn.NLLLoss(reduction="mean")
    # loss_fn = nn.CrossEntropyLoss()
    # best_model = torch.jit.load(os.path.join(snapshot_path, "best-model.pth"))
    #
    # test_loss, test_acc, conf_matrix = eval_single_epoch(fn_loss=loss_fn, test_loader=test_loader, model=best_model)
    # print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    #
    # # # Plot the plots of the learning curves
    # # plt.figure(figsize=(10, 8))
    # # plt.subplot(2, 1, 1)
    # # plt.xlabel("Epoch")
    # # plt.ylabel("NLLLoss")
    # # plt.plot(train_losses, label="train")
    # # plt.plot(test_losses, label="test")
    # # plt.legend()
    # # plt.subplot(2, 1, 2)
    # # plt.xlabel("Epoch")
    # # plt.ylabel("Eval Accuracy [%]")
    # # plt.plot(train_accs, label="train")
    # # plt.plot(test_accs, label="test")
    # # plt.show()
    #
    # # Confusion matrix
    # # more pretty plots using https://github.com/wcipriano/pretty-print-confusion-matrix
    # df_cm = pd.DataFrame(conf_matrix, range(15), range(15))
    # plt.figure(figsize=(20, 14))
    # sn.set(font_scale=1.2) # for label size
    # sn.heatmap(df_cm, fmt='.10g', annot=True, annot_kws={"size": 14}) # font size
    # plt.show()


    os.makedirs(results_path, exist_ok=True)
    try:
        os.remove(loss_acc)
    except OSError as e:
        pass

    unzip_dataset()

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
