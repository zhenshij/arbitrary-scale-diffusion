import torch
import torch.nn as nn

class EncoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_origin, x_recon, split="train"):
        rec_loss = torch.abs(x_origin.contiguous() - x_recon.contiguous()).mean(dim=[1,2,3]).mean()

        log = {
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }

        if split != "train":
            gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [x_origin, x_recon]]
            mse = (gt - pred).pow(2).mean()
            psnr = -10 * torch.log10(mse)
            log.update(
                **{"{}/psnr".format(split): psnr.detach()}
            )

        return rec_loss, log