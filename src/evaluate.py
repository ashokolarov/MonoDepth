import numpy as np
import torch

from dataset import DepthDataset
from model import UNET


def evaluate(model: torch.nn.Module, image: np.array) -> np.array:
    """
    Evaluate trained model on an input image.
    """
    model.eval()

    image = np.expand_dims(image, axis=0)
    image = torch.Tensor(image)

    with torch.no_grad():
        out = model(image)
        out = out.detach().numpy()

    return out


if __name__ == "__main__":
    model_path = "../models/model.pth"
    dataset_path = "../diode/"
    image_size = (256, 256)
    dataset_size = 10

    model = UNET()
    model.load_state_dict(torch.load(model_path))

    dataset = DepthDataset(dataset_path, image_size, dataset_size)

    image, truth = dataset[9]

    pred = evaluate(model, image)
