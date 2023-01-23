import torch
import matplotlib.pyplot as plt
import numpy as np

from model import UNET
from dataset import DepthDataset

device = "cpu"
model_path = "models/model.pth"

if __name__ == "__main__":
    model = UNET().to(device)
    model.load_state_dict(torch.load(model_path))

    path = "/Users/ashokolarov/Documents/Projects/MonoDepth/diode/"
    input_size = (512, 256)
    dataset = DepthDataset(path, input_size)

    img, truth = dataset[0]
    pred = model(torch.Tensor(np.expand_dims(img, axis=0))).detach().numpy()

    pred = pred.reshape(input_size[::-1])
    print(pred.shape)

    plt.imshow(truth.squeeze())
    plt.show()
    plt.imshow(pred)
    plt.show()