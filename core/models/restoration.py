import numpy as np
import torch
import os

from tqdm import tqdm
import torchvision.transforms as transforms
from core.utils.UTILS import compute_psnr, compute_ssim
from core.utils.metrics import calculate_psnr, calculate_ssim


class DoRestoring:
    def __init__(self, restorer, args, config):
        super(DoRestoring, self).__init__()
        self.config = config
        self.restorer = restorer
        self.args = args
        self.not_save_result = self.args.not_save_result
        self.calc_in_Y = self.args.calc_in_Y

        if os.path.isfile(config.resume):
            self.restorer.load_ddm_ckpt(config.resume, ema=True)
            self.restorer.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader):
        sample_folder = os.path.join(self.config.sample_folder, self.config.data.dataset)
        with torch.no_grad():

            with open(os.path.join(self.config.sample_folder, self.config.data.dataset + "_value.txt"), "w") as f:
                num_image = len(val_loader)
                with tqdm(total=num_image) as pbar:
                    ssim_value = 0
                    psnr_value = 0
                    num = 0
                    for i, (x, y) in enumerate(val_loader):
                        y = y[0]
                        print(f"starting processing from image {y}")
                        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                        x = x.to(self.args.device)

                        x_imput = x[:, :-3, :, :]
                        y_gt = x[:, -3:, :, :]

                        y_pred = self.restorer.sample_image(x_imput)
                        y_pred = torch.clamp(y_pred, 0., 1.)

                        num = num + 1
                        if self.calc_in_Y is False:
                            cur_psnr = compute_psnr(y_pred, y_gt)
                            cur_ssim = compute_ssim(y_pred, y_gt)
                        else:
                            y_pred_np = torch.squeeze(y_pred, dim=0).cpu().numpy()
                            y_gt_np = torch.squeeze(y_gt, dim=0).cpu().numpy()

                            y_pred_np = np.transpose(y_pred_np, (1, 2, 0))
                            y_gt_np = np.transpose(y_gt_np, (1, 2, 0))

                            y_pred_np = (y_pred_np * 255).astype(np.uint8)
                            y_gt_np = (y_gt_np * 255).astype(np.uint8)

                            cur_psnr = calculate_psnr(y_pred_np, y_gt_np, test_y_channel=True)
                            cur_ssim = calculate_ssim(y_pred_np, y_gt_np, test_y_channel=True)

                        psnr_value = psnr_value + cur_psnr
                        ssim_value = ssim_value + cur_ssim

                        print(f"this picture value: [{cur_psnr},{cur_ssim}]")
                        f.write(f"image {y} value: [{cur_psnr},{cur_ssim}]\n")
                        print(f"the average value: [{psnr_value / num},{ssim_value / num}]")

                        if self.not_save_result is False:
                            trans_pil = transforms.ToPILImage()
                            y_pred = trans_pil(y_pred[0])
                            if not os.path.exists(sample_folder):
                                os.makedirs(sample_folder)
                            y_pred.save(os.path.join(sample_folder, f"{y}"))

                        pbar.update(1)

                f.write(f"the average value: [{psnr_value / num},{ssim_value / num}]")
                print(f"the average value: [{psnr_value / num},{ssim_value / num}]")

