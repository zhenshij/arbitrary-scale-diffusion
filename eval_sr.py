import argparse
import os
from PIL import Image
import yaml
from functools import partial

import numpy as np

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torchvision import transforms
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.nn.functional import interpolate

from einops import rearrange

import lpips

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import time


# create default evaluator for doctests

def eval_step(engine, batch):
    return batch


default_evaluator = Engine(eval_step)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    model.cuda()
    model.eval()
    return model


def eval_psnr(lr_size, scale_ratio, first_k, eval_type=None, eval_bsize=None, verbose=False, save_image=False, eta=0.0,
              steps=200):
    config = OmegaConf.load(config_path)
    ignore_keys = config.model.params.get('ignore_keys', [])
    ignore_keys.append('loss_fn')
    config.model.params.ignore_keys = ignore_keys
    config.model.params.ckpt_path = ckpt_path

    model = load_model_from_config(config, ckpt_path)

    print('scale_factor:', model.scale_factor)
    output_size = round(lr_size * scale_ratio)

    config.data.params.batch_size = args.batch_size
    config.data.params.train.params.first_k = 1
    config.data.params.validation.params.first_k = first_k
    config.data.params.validation.params.size = output_size

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    loader = data._val_dataloader()

    loss_fn_alex = lpips.LPIPS(net='alex')
    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, 'ssim')

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    vgg.eval()
    vgg = vgg.cuda()

    if eval_type is None:
        metric_fn = calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(calc_psnr, dataset='div2k')
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(calc_psnr, dataset='benchmark')
    else:
        raise NotImplementedError

    psnr_res = Averager()
    lr_psnr_res = Averager()
    ssim_res = Averager()
    lr_ssim_res = Averager()
    lr_mse_res = Averager()
    lpips_res = Averager()
    lr_csim_res = Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    total_image = 0
    total_time = 0.0
    for batch in pbar:

        for k, v in batch.items():
            batch[k] = rearrange(v.cuda(), 'b h w c -> b c h w')

        cond = batch['image_lr']

        b_size = min(args.batch_size, cond.shape[0])

        start = time.time()
        if hasattr(model, 'get_cond'):
            cond = model.get_cond(cond)

        samples, _ = model.sample_log(cond=cond, batch_size=b_size, ddim=True,
                                      ddim_steps=steps, eta=eta, log_every_t=20)

        pred = model.decode_first_stage(samples, output_size=output_size)

        pred = pred * 0.5 + 0.5

        downsampled = interpolate(pred, lr_size)

        pred.clamp_(0, 1)
        downsampled.clamp_(0, 1)

        gt = batch['image_hr']
        gt = gt * 0.5 + 0.5

        lr = batch['image_lr']
        lr = lr * 0.5 + 0.5

        psnr = metric_fn(pred, gt)
        psnr_res.add(psnr.item(), b_size)

        lr_psnr = metric_fn(downsampled, lr)
        lr_psnr_res.add(lr_psnr.item(), b_size)

        state = default_evaluator.run([[pred, gt]])
        ssim_res.add(state.metrics['ssim'], b_size)

        state = default_evaluator.run([[downsampled, lr]])
        lr_ssim_res.add(state.metrics['ssim'], b_size)

        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        pred_n = norm(pred).detach().cpu()
        gt_n = norm(gt).detach().cpu()

        loss_lpips = loss_fn_alex(pred_n, gt_n).mean()
        lpips_res.add(loss_lpips.item(), b_size)

        lr_mse = F.mse_loss(downsampled, lr)
        lr_mse_res.add(lr_mse.item(), b_size)

        q = vgg(interpolate(downsampled, 32))
        k = vgg(interpolate(lr, 32))
        output = cos(q, k).mean()
        lr_csim_res.add(output.item(), b_size)

        if save_image:
            save_image = False
            imgs_path = os.path.join(exp, 'eval_imgs')
            os.makedirs(imgs_path, exist_ok=True)
            for i in range(b_size):
                img_pred = (pred[i].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
                Image.fromarray(img_pred).save(f'{imgs_path}/{i:6d}_pred.png')

                img_lr = (lr[i].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
                Image.fromarray(img_lr).save(f'{imgs_path}/{i:6d}_lr.png')

                img_down = (downsampled[i].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
                Image.fromarray(img_down).save(f'{imgs_path}/{i:6d}_down.png')

        if verbose:
            pbar.set_description('pnsr: {:.4f}, lpips: {:.4f}'.format(psnr_res.item(), lpips_res.item()))

    fin_res = {'PSNR': psnr_res.item(), 'LR_PSNR': lr_psnr_res.item(),
               'SSIM': ssim_res.item(), 'LPIPS': lpips_res.item(),
               'LR_MSE': lr_mse_res.item(), 'LR_SSIM': lr_ssim_res.item(),
               'LR_CSIM': lr_csim_res.item()}

    return fin_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Default
    parser.add_argument('--exp', type=str, required=True, help='Path to the exp')
    parser.add_argument('--lr_size', type=int, required=True, help='The number of sampling images')
    parser.add_argument('--first_k', type=int, default=100, help='The number of sampling images')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of batch size')
    parser.add_argument('--steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--eta', type=float, default=1.0, help='eta of DDIM')
    parser.add_argument('--scale_ratio', type=float, required=True, help='Output size')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Print DDIM progress')
    parser.add_argument('--save_image', type=str2bool, default=False, help='Save outputs')

    args = parser.parse_args()

    exp = args.exp
    exp_data = exp.split('/')[-1].split('_')[0]
    config_path = os.path.join(exp, 'configs', f'{exp_data}-project.yaml')
    ckpt_path = os.path.join(exp, 'checkpoints', 'last.ckpt')

    fin_res = eval_psnr(lr_size=args.lr_size, scale_ratio=args.scale_ratio, first_k=args.first_k, verbose=args.verbose,
                        save_image=args.save_image, eta=args.eta, steps=args.steps)
    print(fin_res)

