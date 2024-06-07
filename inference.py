import os
import sys
import argparse
import yaml
from omegaconf import OmegaConf

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from ldm.util import instantiate_from_config

def load_model(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False) # TODO
    model.cuda()
    model.eval()
    return model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    config_path =f'{args.log_dir}/configs/{log_dir[5:24]}-project.yaml'
    
    if args.epoch < 0:
        ckpt_path = f'{args.log_dir}/checkpoints/last.ckpt'
    else:
        ckpt_path = f'{args.log_dir}/checkpoints/epoch={args.epoch:06d}.ckpt'

    
    config = OmegaConf.load(config_path)

    model = load_model(config, ckpt_path)

    for i in range(0, args.num, args.batch_size):
        bs = (args.num - i) % args.batch_size + 1

    
    samples, _ = model.sample_log(
        batch_size=bs, ddim=True,
        ddim_steps=argssteps, eta=eta, print_bar=False)

def save_img(pred, save_dir, begin=0):
    os.makedirs(save_dir, exist_ok=True)
    pred = pred.detach().cpu()
    pred = ((pred * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
    for i in range(pred.shape[0]):
        img = Image.fromarray(pred[i].numpy())
        save_path = os.path.join(save_dir, f'{begin + i:06d}.png')
        img.save(save_path)


def main(args):
    exp = args.log_dir.split('/')[-1]
    config_path = f'{args.log_dir}/configs/{exp[:19]}-project.yaml'
    save_dir = f'{args.save_dir}/{exp}'
    
    if args.epoch < 0:
        ckpt_path = f'{args.log_dir}/checkpoints/last.ckpt'
    else:
        ckpt_path = f'{args.log_dir}/checkpoints/epoch={args.epoch:06d}.ckpt'

    
    config = OmegaConf.load(config_path)

    model = load_model(config, ckpt_path)

    pbar = tqdm(range(0, args.num, args.batch_size), total=args.num, leave=False, desc='inference')
    for i in pbar:
        bs = min((args.num - i), args.batch_size)
    
        samples, _ = model.sample_log(
            batch_size=bs, ddim=True, cond=None,
            ddim_steps=args.steps, eta=args.eta, print_bar=args.verbose)

        for s in args.size:
            pred = model.decode_first_stage(samples, output_size=int(s))
            save_path = os.path.join(save_dir, f'{s}')
            save_img(pred, save_path, i)

        pbar.update(bs)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Default
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the exp')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the save images')
    parser.add_argument('--epoch', type=int, default=-1, help='Epoch')
    parser.add_argument('--num', type=int, default=50000, help='The number of sampling images')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of batch size')
    parser.add_argument('--steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--eta', type=float, default=1.0, help='eta of DDIM')
    parser.add_argument('--size', nargs='+', required=True, help='Output size')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Print DDIM progress')

    args = parser.parse_args()

    main(args)

