import logging
import os
import time
import random

from torch import nn
from torchvision import transforms, models
from tqdm import tqdm
import torch
from core.models.backbone import HSSNet
from core.utils.loss_function import DepthLoss, LossNetwork
from core.utils.pytorch_ssim import ssim
from core.utils.optimizer import get_optimizer
from torch.nn import ReflectionPad2d
from core.utils import scheduler as lr_scheduler
import torch.nn.functional as F


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)


def padding(img_lq, patch_size, patch_overlap):
    _, _, h, w = img_lq.shape
    stride = patch_size - patch_overlap

    h_pad = (patch_overlap - h) % stride
    w_pad = (patch_overlap - w) % stride

    if h_pad == 0 and w_pad == 0:
        return img_lq

    pad = ReflectionPad2d((0, w_pad, 0, h_pad))
    img_lq = pad(img_lq)

    return img_lq


def patch_mask(patch_size, patch_overlap):
    width, height = patch_size, patch_size
    center_x, center_y = (width - 1) / 2, (height - 1) / 2

    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    x, y = torch.meshgrid(x, y, indexing='ij')

    dist = center_x - patch_overlap
    distance_squared = torch.clamp((torch.maximum(torch.abs(x - center_x), torch.abs(y - center_y)) - dist), 0.0)

    result = distance_squared.max() - distance_squared + 1
    result = result / result.max()

    return result


class MixAugment:
    def __init__(self, device, dataset_mixup_prob=0.1):
        self.dataset_dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
        self.dataset_mixup_prob = dataset_mixup_prob

        self.device = device

        self.augments = [self.dataset_mixup]

    def dataset_mixup(self, lq, gt):
        lam = self.dataset_dist.rsample((1, 1)).item()

        r_index = torch.randperm(gt.size(0)).to(self.device)

        gt = lam * gt + (1 - lam) * gt[r_index, :]
        lq = lam * lq + (1 - lam) * lq[r_index, :]

        return lq, gt

    def __call__(self, lq, gt):

        # mixup data augments
        dataset_mixup_random = random.random()
        if dataset_mixup_random < self.dataset_mixup_prob:
            augment = random.randint(0, len(self.augments) - 1)
            lq, gt = self.augments[augment](lq, gt)

        return lq, gt


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class Restoration(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = args.device

        self.model = HSSNet.Backbone(config)

        self.model.to(self.device)
        if args.parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = get_optimizer(self.config, self.model.parameters())
        self.scheduler = lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer=self.optimizer,
                                                                     periods=self.config.scheduler.periods,
                                                                     restart_weights=self.config.scheduler.restart_weights,
                                                                     eta_mins=self.config.scheduler.eta_mins)
        self.start_epoch, self.step = 0, 0

        self.patch_size = config.backbone.image_size
        self.patch_overlap = config.sampling.patch_overlap
        self.sample_patch_num = config.sampling.patch_num
        self.sample_patch_size = config.sampling.patch_size

        self.mixup_flag = config.data_augment.augment
        self.mixup_prob = config.data_augment.mixup_prob
        self.mixup_augmentation = MixAugment(self.device, self.mixup_prob)
        self.pixel_l1_weight = float(0.0)

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = load_checkpoint(load_path, self.device)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.pixel_l1_weight = checkpoint['pixel_l1_weight']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        tb_loss = self.args.tb_loss
        tb_pixell1weight = self.args.tb_pixell1weight

        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.config.resume):
            print("restoring network params")
            self.load_ddm_ckpt(self.config.resume)

        n_epochs = self.config.training.step_size * self.config.training.batch_size // self.config.data.train_num + 1

        if self.config.loss.perceptual_weight != 0.0:
            vgg = models.vgg16(pretrained=False)
            vgg.load_state_dict(torch.load('../pretrain_ckpts/vgg16-397923af.pth'))
            vgg_model = vgg.features[:16]
            vgg_model = vgg_model.to(self.device)
            for param in vgg_model.parameters():
                param.requires_grad = False
            perceptual_calculate = LossNetwork(vgg_model)
            perceptual_calculate.eval()

        if self.config.loss.depth_weight != 0.0:
            depth_calculate = DepthLoss(device=self.device)

        print(f"loss weight:\n"
              f"depth_weight:{self.config.loss.depth_weight}\n"
              f"perceptual_weight:{self.config.loss.perceptual_weight}\n"
              f"ssim_weight:{self.config.loss.ssim_weight}\n"
              f"pixel_weight:{self.config.loss.pixel_weight}\n")

        with tqdm(total=(self.config.training.step_size - self.step)) as pbar:
            print(f"total epoch: {n_epochs - self.start_epoch}")
            for epoch in range(self.start_epoch, n_epochs):
                if self.step >= self.config.training.step_size:
                    break
                print(f"epoch:{epoch}--step:{self.step}")

                data_start = time.time()
                data_time = 0

                for i, (x, y) in enumerate(train_loader):
                    if self.step >= self.config.training.step_size:
                        break
                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x

                    data_time += time.time() - data_start

                    self.model.train()
                    self.step += 1

                    x = x.to(self.device)

                    # get lq and gt
                    x_input = x[:, :-3, :, :]
                    y_gt = x[:, -3:, :, :]

                    # mixing_augmentation
                    if self.mixup_flag and self.mixup_prob > 0:
                        x_input, y_gt = self.mixup_augmentation(x_input, y_gt)

                    loss_weight = [
                        self.config.loss.depth_weight,
                        self.config.loss.perceptual_weight,
                        self.config.loss.ssim_weight,
                        self.config.loss.pixel_weight
                    ]

                    y_pred = self.model(x=x_input)

                    if self.config.loss.perceptual_weight != 0.0:
                        perceptual_loss = perceptual_calculate(y_pred, y_gt)
                    else:
                        perceptual_loss = 0.0

                    if self.config.loss.depth_weight != 0.0:
                        depth_loss = depth_calculate(y_pred, y_gt)
                    else:
                        depth_loss = 0.0

                    if self.config.loss.depth_weight != 0.0:
                        ssim_loss = 1 - ssim(y_pred, y_gt)
                    else:
                        ssim_loss = 0.0

                    smooth_l1_loss = F.smooth_l1_loss(y_pred, y_gt)
                    l1_loss = F.l1_loss(y_pred, y_gt)

                    with torch.no_grad():
                        self.pixel_l1_weight = 0.999 * self.pixel_l1_weight + 0.001 * min(l1_loss, 1)
                        pixel_l1_weight = self.pixel_l1_weight
                    pixel_loss = smooth_l1_loss + pixel_l1_weight * l1_loss

                    total_loss = loss_weight[0] * depth_loss + \
                                 loss_weight[1] * perceptual_loss + \
                                 loss_weight[2] * ssim_loss + \
                                 loss_weight[3] * pixel_loss

                    tb_loss.add_scalar("loss", total_loss, global_step=self.step)
                    tb_pixell1weight.add_scalar("pixel_l1_weight", self.pixel_l1_weight, global_step=self.step)

                    losses = [total_loss.item(), self.pixel_l1_weight]

                    if self.step % 10 == 0:
                        logging.info(
                            f"step: {self.step}, loss: {total_loss.item()}, data time: {data_time / (i + 1)}"
                        )

                    if self.step % 1000 == 0:
                        print(f"step: {self.step}, loss: {losses}, data time: {data_time / (i + 1)}")

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    self.ema_helper.update(self.model)

                    self.scheduler.step()
                    data_start = time.time()

                    if self.step % self.config.training.validation_freq == 0:
                        self.model.eval()
                        self.sample_validation(val_loader, self.step)

                    if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                        if self.step != 1:
                            pbar.update(self.config.training.snapshot_freq)
                        print(f"save_checkpoint:{self.config.data.dataset + str(self.step)}")
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        save_checkpoint({
                            'epoch': epoch,
                            'step': self.step,
                            'pixel_l1_weight': self.pixel_l1_weight,
                            'state_dict': model_to_save.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'config': self.config
                        }, filename=os.path.join(self.config.data.model_param_dir, 'ckpts',
                                                 self.config.data.dataset + str(self.step)))

    def sample_image(self, x_imput):

        b, c, h, w = x_imput.shape

        if self.args.no_patch is False:
            patch_size = self.sample_patch_size
            patch_overlap = self.patch_overlap
            patch_num = self.sample_patch_num

            out_patch_mask = patch_mask(patch_size, patch_overlap).to(self.device)

            # padding
            x_imput = padding(x_imput, patch_size, patch_overlap)
            # patch
            stride = patch_size - patch_overlap
            h_idx_list = list(range(0, h - patch_overlap, stride))
            w_idx_list = list(range(0, w - patch_overlap, stride))
            E = torch.zeros(x_imput.shape).type_as(x_imput)
            W = torch.zeros_like(E)

            with torch.no_grad():
                in_patches = []
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = x_imput[..., h_idx:h_idx + patch_size, w_idx:w_idx + patch_size]
                        in_patches.append(in_patch)

                tensors = []
                start_idx = 0
                while start_idx < len(in_patches):
                    end_idx = min(start_idx + patch_num, len(in_patches))
                    tensor = torch.concat(in_patches[start_idx:end_idx])
                    tensors.append(tensor)
                    start_idx = end_idx

                out_patches = []
                for tensor in tensors:
                    out_tensor = self.model(tensor)
                    num, _, _, _ = out_tensor.shape
                    for id in range(num):
                        out_patches.append(out_tensor[id:id + 1, :, :, :])

                patch_id = 0
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        E[..., h_idx:(h_idx + patch_size), w_idx:(w_idx + patch_size)].add_(out_patches[patch_id] * out_patch_mask)
                        W[..., h_idx:(h_idx + patch_size), w_idx:(w_idx + patch_size)].add_(out_patch_mask)
                        patch_id += 1

                y_pred = E.div_(W)

            # Unpad the output
            y_pred = y_pred[:, :, :h, :w]
        else:
            mini_window = 16
            h_pad = (-h) % mini_window
            w_pad = (-w) % mini_window
            if h_pad != 0 or w_pad != 0:
                pad = ReflectionPad2d((0, w_pad, 0, h_pad))
                x_imput = pad(x_imput)
            y_pred = self.model(x_imput)[:, :, :h, :w]
        return y_pred

    def sample_validation(self, val_loader, step):
        val_folder = os.path.join(self.config.val_folder,
                                  self.config.data.dataset + str(self.config.backbone.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                if x.ndim == 5:
                    x = x.flatten(start_dim=0, end_dim=1)
                else:
                    x = x
                y = y[0]
                y = os.path.splitext(y)
                y = y[0]

                n = x.size(0)
                x_imput = x[:, :-3, :, :].to(self.device)
                y_gt = x[:, -3:, :, :].to(self.device)

                y_pred = self.sample_image(x_imput)
                y_pred = torch.clamp(y_pred, 0., 1.)
                trans_pil = transforms.ToPILImage()

                for i in range(n):
                    imput = trans_pil(x_imput[0])
                    gt = trans_pil(y_gt[0])
                    pred = trans_pil(y_pred[0])
                    if not os.path.exists(os.path.join(val_folder, str(step))):
                        os.makedirs(os.path.join(val_folder, str(step)))
                    imput.save(os.path.join(val_folder, str(step), f"{y}_cond.png"))
                    gt.save(os.path.join(val_folder, str(step), f"{y}_gt.png"))
                    pred.save(os.path.join(val_folder, str(step), f"{y}.png"))
