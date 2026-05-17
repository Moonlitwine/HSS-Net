import torch
from torch import nn
import torch.nn.functional as F

from core.utils import depth_network


def RGB2YCrCb(input_im, device):
    if str(device) == "cuda":
        device = torch.device("cuda:0")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


# Implementation of perceptual loss introduced in: 'Learning Weather-General and Weather-Specific Features for Image
# Restoration Under Multiple Adverse Weather Conditions'(Zhu et al., 2023) https://github.com/zhuyr97/WGWS-Net
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)


# Implementation of depth loss introduced in: 'Learning Weather-General and Weather-Specific Features for Image
# Restoration Under Multiple Adverse Weather Conditions'(Zhu et al., 2023) https://github.com/zhuyr97/WGWS-Net
class DepthLoss(nn.Module):
    def __init__(self, device):
        super(DepthLoss, self).__init__()
        self.criterion = nn.L1Loss()

        self.encoder = depth_network.ResnetEncoder(18, False)
        encoder_path = '../pretrain_ckpts/encoder.pth'
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        self.depth_decoder = depth_network.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                                        scales=range(4))
        depth_decoder_path = '../pretrain_ckpts/depth.pth'
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(device)
        self.depth_decoder.eval()

    def forward(self, x, y):
        x_in = F.interpolate(x,size=(self.feed_height,self.feed_width),mode='bilinear')
        y_in = F.interpolate(y,size=(self.feed_height,self.feed_width),mode='bilinear')

        x_out = self.depth_decoder(self.encoder(x_in, 'day', 'val'))
        y_out = self.depth_decoder(self.encoder(y_in, 'day', 'val'))
        x_out = x_out[("disp", 0)]
        y_out = y_out[("disp", 0)]

        return self.criterion(x_out, y_out)
