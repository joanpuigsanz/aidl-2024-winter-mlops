import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    pred = predicted_batch.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum
    

def save_model(model, path):
    torch.save(model.state_dict(), path)

def show_loader_images(data_set, data_loader):
    # Check loader data
    print(f"Image set size = {len(data_set)}")
    img, label = data_set[0]
    print("Img shape: ", img.shape)
    print("Label: ", label)

    # Print one image
    plt.imshow(img.squeeze())
    plt.show()

    # Similarly, we can sample a BATCH from the dataloader by running over its iterator
    iter_ = iter(data_loader)
    bimg, blabel = next(iter_)
    print("Batch Img shape: ", bimg.shape)
    print("Batch Label shape: ", blabel.shape)

    # And now let's look at the kind of images we are dealing with

    # make_grid is a function from the torchvision package that transforms a batch
    # of images to a grid of images
    img_grid = make_grid(bimg)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid.permute(1, 2, 0), interpolation="nearest")
    plt.show()
