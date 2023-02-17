import torch
from tqdm import tqdm

from dataset import instantiate_loader
from model import UNET
from utils import DepthLoss

LR = 1E-3
dataset_path = "../diode/"
dataset_size = 80
image_size = (256, 256)
batch_size = 5
epochs = 5
device = "mps"


def train_func(model: torch.nn.Module, optimizer: torch.optim,
               loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module) -> torch.nn.Module:

    loop = tqdm(loader)
    num_batches = len(loader)

    epoch_average_loss = 0.

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        preds = model(data)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_average_loss += loss.item() / num_batches

        loop.set_postfix({"loss": epoch_average_loss})

    return model


if __name__ == "__main__":
    model = UNET().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_loader = instantiate_loader(dataset_path,
                                     image_size,
                                     batch_size,
                                     training=True,
                                     dataset_size=dataset_size)

    loss = DepthLoss([0.85, 0.7, 0.5]).to(device=device)

    for epoch in range(epochs):
        train_func(model, optimizer, data_loader, loss)

    torch.save(model.state_dict(), "../models/model.pth")