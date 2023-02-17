import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import utils


class DepthDataset(Dataset):
    """
    Create a Dataset instance from the DIODE dataset. 
    """

    def __init__(self,
                 dataset_path: str,
                 image_size: Tuple[int, int],
                 dataset_size: int = None) -> None:
        """
        Class constructor used to create an instance.

        Arguments
        ---------
        dataset_path
            path to downloaded DIODE datset
        input_size
            dictates the size to which the images are resized - (Width, Height)
        N
            number of images in the dataset to extract. Defaults to the full dataset.
        """

        self._dataset_path = dataset_path
        self._size = image_size
        self._N = dataset_size

        self._files = self.walk_directory(dataset_path)

        self._data_paths = {
            "image": [x for x in self._files if x.endswith(".png")],
            "depth": [x for x in self._files if x.endswith("_depth.npy")],
            "mask": [x for x in self._files if x.endswith("_depth_mask.npy")]
        }

        self._min_depth = 0.5

        self._initialize_dataset()

        self._transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    @staticmethod
    def diode_download_and_extract(path: str) -> None:
        """
        Download the DIODE dataset to a predefined path

        Arguments
        ---------
        path
            directory to which to save the dataset
        """
        url = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"
        filename = "diode.tar.gz"
        utils.download_and_extract_archive(url, path, filename)

    @staticmethod
    def walk_directory(dir: str) -> List[str]:
        """
        Walk a directory and record all the files present in it.

        Arguments
        ---------
        dir
            path to directory
        """
        item_list = os.listdir(dir)

        file_list = list()
        for item in item_list:
            item_dir = os.path.join(dir, item)
            if os.path.isdir(item_dir):
                file_list += DepthDataset.walk_directory(item_dir)
            else:
                file_list.append(item_dir)
        return sorted(file_list)

    def _initialize_dataset(self):
        if self._N is None:
            self._N = len(self._data_paths['image'])

        self.images = np.zeros((self._N, *reversed(self._size), 3),
                               dtype=np.float32)
        self.depths = np.zeros((self._N, *reversed(self._size), 1),
                               dtype=np.float32)

        for idx in range(self._N):
            image_path = self._data_paths['image'][idx]
            depth_path = self._data_paths['depth'][idx]
            mask_path = self._data_paths['mask'][idx]

            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self._size, cv2.INTER_LINEAR)

            depth = np.load(depth_path).squeeze()

            mask = np.load(mask_path)
            mask = mask > 0

            max_depth = min(300, np.percentile(depth, 99))
            depth = np.clip(depth, self._min_depth, max_depth)
            depth = np.log(depth, where=mask)

            depth = np.ma.masked_where(~mask, depth)

            depth = np.clip(depth, 0.1, np.log(max_depth))
            depth = cv2.resize(depth, self._size, cv2.INTER_LINEAR)

            self.images[idx, :, :, :] = image
            self.depths[idx, :, :, 0] = depth

    def __len__(self) -> int:
        """
        Return size of the dataset.
        """
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate an input/output pair from the dataset.

        Arguments
        ---------
        idx
            index of the desired pair
        """
        img = self._transforms(self.images[idx])
        depth = self._transforms(self.depths[idx])

        return img, depth


def instantiate_loader(dataset_path: str,
                       image_size: Tuple[int, int],
                       batch_size: int,
                       training: bool,
                       dataset_size: int = None,
                       **kwargs):
    """
    Generate a dataloader object used for training.

    Arguments
    ---------
    dataset_path
        path to directory where the dataset resides
    image_size
        size to which to resize the images in the dataset
    batch_size
        size of the a single batch
    training
        indicates whether the instance is used for training purposes
    dataset_size
        si
    """
    dataset = DepthDataset(dataset_path, image_size, dataset_size)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if training else False,
                      **kwargs)


if __name__ == "__main__":
    path = "/Users/ashokolarov/Documents/Projects/MonoDepth/diode/"
    image_size = (256, 256)

    dataset = DepthDataset(path, image_size)

    img, depth = dataset[0]
