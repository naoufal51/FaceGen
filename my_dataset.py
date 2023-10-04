import os
from typing import Callable

from PIL import Image

from torch.utils.data import Dataset

import torch


class DatasetDirectory(Dataset):
    """
    A custom dataset class that loads images from folder.
    args:
    - directory: location of the images
    - transform: transform function to apply to the images
    - extension: file format
    """

    def __init__(
        self, directory: str, transforms: Callable = None, extension: str = ".jpg"
    ):
        self.transforms = transforms
        self.images = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(extension)
        ]

    def __len__(self) -> int:
        """returns the number of items in the dataset"""
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        """load an image and apply transformation"""
        image_path = self.images[index]

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        return image
