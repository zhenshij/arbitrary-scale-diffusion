import random
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange

from ldm.modules.diffusionmodules.model import Encoder, Decoder, LIIF
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import make_coord_cell, to_pixel_samples

def disabled_train(self, mode=True):
    return self


class IND(nn.Module):
    def __init__(self, ddconfig, liifconfig):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1),
            Decoder(**ddconfig)
        )
        self.inr = LIIF(in_dim=ddconfig['ch']*ddconfig['ch_mult'][0], out_dim=3, **liifconfig)

    def forward(self, z, coord=None, cell=None, output_size=None, return_img=True, bsize=0):
        h = self.decoder(z)
        return self.inr(h, coord=coord, cell=cell, output_size=output_size)


class FirstStageModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 liifconfig,
                 lossconfig,
                 trainconfig=None,
                 valconfig=None,
                 scheduler_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 nll_weight=0.0,
                 use_saconv=False,
                 freeze_encoder=False,
                 ):
        super().__init__()
        self.trainconfig = trainconfig
        self.valconfig = valconfig

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.encoder = nn.Sequential(
            Encoder(**ddconfig),
            nn.Conv2d(2*ddconfig["z_channels"], 2*ddconfig["z_channels"], 1) if ddconfig["double_z"] \
            else nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        )

        self.decoder = IND(ddconfig=ddconfig, liifconfig=liifconfig)

        self.loss = instantiate_from_config(lossconfig)
        self.use_posterior = ddconfig["double_z"]

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze_encoder:
            self.encoder = self.encoder.eval()
            self.encoder.train = disabled_train
            for param in self.encoder.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif 'model' in ckpt:
            sd = torch.load(path, map_location="cpu")["model"]['sd']
        else:
            raise NotImplementedError

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # For div2k begin
        # for k in keys:
        #     if k.startswith('encoder'):
        #         sd['encoder.0' + k[7:]] = sd[k]
        #         del sd[k]
        #     elif k.startswith('quant_conv'):
        #         sd['encoder.1' + k[10:]] = sd[k]
        #         del sd[k]
        # end
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        if self.use_posterior:
            return DiagonalGaussianDistribution(h)
        else:
            return h

    def decode(self, z, coord=None, cell=None, output_size=None, return_img=True, bsize=0):
        return self.decoder(z, coord=coord, cell=cell, output_size=output_size, return_img=return_img, bsize=bsize)

    def forward(self, input, coord=None, cell=None, output_size=None, 
                return_img=True, bsize=0, sample_posterior=True):
        posterior = self.encode(input)
        if not self.use_posterior:
            z = posterior
        elif sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, coord, cell, output_size, return_img, bsize)
        return dec, posterior

    def get_input(self, batch, config):
        gt_min = config['gt_min']
        gt_max = config['gt_max']
        gt_size = int(random.uniform(gt_min, gt_max))

        inp = batch[config['image_key']]
        inp = rearrange(inp, 'b h w c -> b c h w')
        gt = F.interpolate(inp, gt_size, mode='bicubic')

        inp = inp.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()

        fconfig = {'bsize': config.get('bsize', 0)}
        if 'sample_q' in config:
            sample_q = config['sample_q']
            coord, cell, gt = to_pixel_samples(gt)
            sample_lst = np.random.choice(
                len(coord), sample_q, replace=False)
            coord = coord[sample_lst]
            cell = cell[sample_lst]
            gt = gt[sample_lst]

            fconfig.update(
                coord=coord,
                cell=cell
            )
        else:
            fconfig.update(
                output_size=gt.shape[-1],
            )
        return inp, gt, fconfig

    def training_step(self, batch, batch_idx):
        inp, gt, fconfig = self.get_input(batch, self.trainconfig)
        reconstructions, posterior = self(inp, **fconfig)

        rec_loss, log_dict = self.loss(gt, reconstructions, split="train")
        self.log("rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return rec_loss

    def validation_step(self, batch, batch_idx):
        inp, gt, fconfig = self.get_input(batch, self.valconfig)
        reconstructions, posterior = self(inp, **fconfig)
        rec_loss, log_dict = self.loss(gt, reconstructions, split="val")

        self.log("val/rec_loss", log_dict["val/rec_loss"])
        self.log_dict(log_dict)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                ]
            return [opt], scheduler

        return opt

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inp, gt, fconfig = self.get_input(batch, self.valconfig) # TODO
        inp = inp.to(self.device)
        if not only_inputs:
            xrec, posterior = self(inp, **fconfig)
            log["reconstructions"] = xrec
            log["gt"] = gt
        log["inputs"] = inp
        return log
