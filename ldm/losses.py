import torch
import lpips


loss_fn_alex = lpips.LPIPS(net='alex').cuda()
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def cosine_similarity_loss(pred, target):
    return 1 - torch.nn.functional.cosine_similarity(target, pred)


def lpips_alex_loss(pred, target):
    return loss_fn_alex(pred, target)


def lpips_vgg_loss(pred, target):
    return loss_fn_vgg(pred, target)
