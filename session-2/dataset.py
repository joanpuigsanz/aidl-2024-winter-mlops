import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(
        self,
        images_path,
        labels_path,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    ):
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        image_path = os.path.join(
            self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg"
        )
        image = Image.open(image_path)
        return (
            self.transform(image),
            code - 1,
        )  # code -1 will make the code to start from 0 instead of 1


if __name__ == "__main__":
    dataset = MyDataset(
        os.path.join("data", "data", "data"),
        os.path.join("data", "chinese_mnist.csv"),
    )
    print(len(dataset))
    print(dataset[0])
