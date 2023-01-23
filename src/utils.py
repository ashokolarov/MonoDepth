import torch
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional import image_gradients


class DepthLoss(torch.nn.Module):

    def __init__(self, weights):
        super(DepthLoss, self).__init__()
        self.weights = weights

        self._SSIM = SSIM(reduction="elementwise_mean")
        self._L1 = torch.nn.L1Loss(reduction="mean")

    @staticmethod
    def disparity_smoothness(preds: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        dy_preds, dx_preds = image_gradients(preds)
        dy_target, dx_target = image_gradients(target)

        wx = torch.exp(torch.mean(torch.abs(dx_target)))
        wy = torch.exp(torch.mean(torch.abs(dy_target)))

        sx = dx_preds * wx
        sy = dy_preds * wy

        loss = torch.mean(torch.abs(sx)) + torch.mean(torch.abs(sy))

        return loss

    def forward(self, preds: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        SSIM = self._SSIM(preds, target)
        L1 = self._L1(preds, target)
        DSL = self.disparity_smoothness(preds, target)

        return self.weights[0] * SSIM + self.weights[1] * L1 + self.weights[
            2] * DSL


if __name__ == "__main__":
    loss = DepthLoss([0.85, 0.7, 0.5])

    input = torch.randn(5, 3, 50, 50)
    target = torch.randn(5, 3, 50, 50)

    print(loss.forward(input, target))