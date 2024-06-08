import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .unet_arch import UNet
from .arch_util import flow_warp, ConvResidualBlocks, SmallUpdateBlock
from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class EvTexture(nn.Module):
    """EvTexture: Event-driven Texture Enhancement for Video Super-Resolution (ICML 2024)
       Note that: this class is for 4x VSR

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 30
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # RGB-based flowalignment
        self.spynet = SpyNet(spynet_path)
        self.cnet = ConvResidualBlocks(num_in_ch=3, num_out_ch=64, num_block=8)

        # iterative texture enhancement module
        self.enet = UNet(inChannels=1, outChannels=num_feat)
        self.update_block = SmallUpdateBlock(hidden_dim=num_feat, input_dim=num_feat * 2)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)

        # propogation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat * 2 + 3, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    # context feature extractor
    def get_feat(self, x):
        b, n, c, h, w = x.size()
        feats_ = self.cnet(x.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h, w)

        return feats_

    def forward(self, imgs, voxels_f, voxels_b):
        """Forward function of EvTexture

        Args:
            imgs: Input frames with shape (b, n, c, h, w). n is the number of frames.
            voxels_f: forward event voxel grids with shape (b, n-1 , c, h, w).
            voxels_b: backward event voxel grids with shape (b, n-1 , c, h, w).

        Output:
            out_l: output frames with shape (b, n, c, 4h, 4w)
        """

        flows_forward, flows_backward = self.get_flow(imgs)
        feat_imgs = self.get_feat(imgs)
        b, n, _, h, w = imgs.size()
        bins = voxels_f.size()[2]

        # backward branch
        out_l = []
        feat_prop = imgs.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = imgs[:, i, :, :, :]

            if i < n - 1:
                # motion branch by rgb frames
                flow = flows_backward[:, i, :, :, :]
                feat_prop_coarse = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                # texture branch by event voxels
                hidden_state = feat_prop.clone()
                feat_img = feat_imgs[:, i, :, :, :]  # [B, num_feat, H, W]
                cur_voxel = voxels_f[:, i, :, :, :]  # [B, Bins, H, W]

                ## iterative update block
                feat_prop_fine = feat_prop.clone()
                for j in range(bins - 1, -1, -1):
                    voxel_j = cur_voxel[:, j, :, :].unsqueeze(1)  # [B, 1, H, W]
                    feat_motion = self.enet(voxel_j)  # [B, num_feat, H, W], enet is UNet(inChannels=1, OurChannels=num_feat)
                    hidden_state, delta_feat = self.update_block(hidden_state, feat_img, feat_motion)  # refine coarse hidden state
                    feat_prop_fine = feat_prop_fine + delta_feat

                feat_prop = self.fusion(torch.cat([feat_prop_fine, feat_prop_coarse], dim=1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = imgs[:, i, :, :, :]

            if i > 0:
                # motion branch by rgb frames
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop_coarse = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                # texture branch by event voxels
                hidden_state = feat_prop.clone()
                feat_img = feat_imgs[:, i, :, :, :]  # [B, num_feat, H, W]
                cur_voxel = voxels_b[:, i - 1, :, :, :]  # [B, Bins, H, W]

                # iterative update block
                feat_prop_fine = feat_prop.clone()
                for j in range(bins - 1, -1, -1):
                    voxel_j = cur_voxel[:, j, :, :].unsqueeze(1)  # [B, 1, H, W]
                    feat_motion = self.enet(voxel_j)  # [B, num_feat, H, W], enet is UNet(inChannels=1, OurChannels=64)
                    hidden_state, delta_feat = self.update_block(hidden_state, feat_img, feat_motion)
                    feat_prop_fine = feat_prop_fine + delta_feat

                feat_prop = self.fusion(torch.cat([feat_prop_fine, feat_prop_coarse], dim=1))

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)