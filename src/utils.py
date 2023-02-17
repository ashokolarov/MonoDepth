import torch
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.functional import image_gradients


class DepthLoss(torch.nn.Module):

    def __init__(self, weights):
        """
        Generate DepthLoss instance.

        Arguments
        ---------
        weights
            weights assigned to the different loss functions. 
            weights[0] - weight corresponding to SSIM
            weights[1] - weight corresponding to L1
            weights[2] - weight corresponding to Disparity smoothness
        """
        super(DepthLoss, self).__init__()
        self._wSSIM, self._wL1, self._wDS = weights

        self._SSIM = SSIM()
        self._L1 = torch.nn.L1Loss()

    @staticmethod
    def disparity_smoothness(preds: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        dy_preds, dx_preds = image_gradients(preds)
        dy_target, dx_target = image_gradients(target)

        wx = torch.exp(-torch.mean(torch.abs(dx_target)))
        wy = torch.exp(-torch.mean(torch.abs(dy_target)))

        sx = dx_preds * wx
        sy = dy_preds * wy

        loss = torch.mean(torch.abs(sx)) + torch.mean(torch.abs(sy))

        return loss

    def forward(self, preds: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        SSIM = 1 - self._SSIM(preds, target)
        L1 = self._L1(preds, target)
        DS = self.disparity_smoothness(preds, target)

        return self._wSSIM * SSIM + self._wL1 * L1 + self._wDS * DS


if __name__ == "__main__":
    loss = DepthLoss([0.85, 0.7, 0.5])

    input = torch.randn(10, 3, 50, 50)
    target = torch.randn(10, 3, 50, 50)

    loss_value = loss.forward(input, target)
    print(loss_value)