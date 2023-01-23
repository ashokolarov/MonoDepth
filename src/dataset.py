import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import utils


class DepthDataset(Dataset):

    def __init__(self, dataset_path: str, input_size: Tuple[int, int]):
        self._dataset_path = dataset_path
        self._size = input_size

        self._files = self.walk_directory(dataset_path)

        self._data_paths = {
            "image": [x for x in self._files if x.endswith(".png")],
            "depth": [x for x in self._files if x.endswith("_depth.npy")],
            "mask": [x for x in self._files if x.endswith("_depth_mask.npy")]
        }

        self._min_depth = 0.5

        self._initialize_dataset()

    @staticmethod
    def diode_download_and_extract(url: str, path: str) -> None:
        filename = "diode.tar.gz"
        utils.download_and_extract_archive(url, path, filename)

    @staticmethod
    def walk_directory(dir: str) -> List[str]:
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
        N = 50  # len(self._data_paths['image'])

        self.images = np.zeros((N, 3, *reversed(self._size)), dtype=np.float32)
        self.depths = np.zeros((N, 1, *reversed(self._size)), dtype=np.float32)

        for idx in range(N):
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

            image = image.transpose((2, 0, 1))

            self.images[idx, :, :, :] = image
            self.depths[idx, 0, :, :] = depth

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.Tensor(self.images[idx])
        depth = torch.Tensor(self.depths[idx])

        return img, depth


def instantiate_loader(dataset_path: str, input_size: Tuple[int, int],
                       batch_size: int, training: bool, **kwargs):
    dataset = DepthDataset(dataset_path, input_size)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if training else False,
                      **kwargs)


if __name__ == "__main__":
    url = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"
    path = "/Users/ashokolarov/Documents/Projects/MonoDepth/diode/"

    input_size = (1024, 768)

    dataset = DepthDataset(path, input_size)

    img, depth = dataset[0]
