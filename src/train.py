import torch
from tqdm import tqdm

from model import UNET
from dataset import instantiate_loader
from utils import DepthLoss

LR = 1E-4
dataset_path = "/Users/ashokolarov/Documents/Projects/MonoDepth/diode/"
input_size = (512, 256)
batch_size = 5
epochs = 2
device = "mps"


def train_func(model, optimizer, loader, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        preds = model(data)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model = UNET().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_loader = instantiate_loader(dataset_path,
                                     input_size,
                                     batch_size,
                                     training=True)

    loss = DepthLoss([0.85, 0.7, 0.5]).to(device=device)

    for epoch in range(epochs):
        train_func(model, optimizer, data_loader, loss)

    torch.save(model.state_dict(), "models/model.pth")
