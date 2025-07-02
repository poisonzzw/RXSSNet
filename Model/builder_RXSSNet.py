from Convnextv2.convnextv2 import ConvNeXtV2
from Model.crossFormer_token import MHCAFormer, Token_corrector
import torch
import torch.nn.functional as F
import torch.nn as nn
from Model.CSAFM import CSFNet as csf

class model(nn.Module):
    def __init__(self, in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs):
        super().__init__()
        self.num_stages = 4
        self.encoder_rgb = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)
        self.encoder_thermal = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)

        input_linear_resolution = [19200, 4800, 1200, 300] # [H/4 * W/4, ... , H/32 * W/32]

        # CGPI
        self.Token_corrector = nn.ModuleList([
            Token_corrector(stride=4, patch_size=7, in_chans=6, embed_dim=dims[0], input_resolution=input_linear_resolution[0]),
            Token_corrector(stride=8, patch_size=7, in_chans=6, embed_dim=dims[1], input_resolution=input_linear_resolution[1]),
            Token_corrector(stride=16, patch_size=7, in_chans=6, embed_dim=dims[2], input_resolution=input_linear_resolution[2]),
            Token_corrector(stride=32, patch_size=7, in_chans=6, embed_dim=dims[3], input_resolution=input_linear_resolution[3]),
            ])

        # CRM
        self.csf = nn.ModuleList([
            csf(channels=dims[0]),
            csf(channels=dims[1]),
            csf(channels=dims[2]),
            csf(channels=dims[3])
        ])

        # LAM
        self.crossFormer = nn.ModuleList([
            MHCAFormer(patch_size=7, dim=dims[0], reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d),
            MHCAFormer(patch_size=3, dim=dims[1], reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d),
            MHCAFormer(patch_size=3, dim=dims[2], reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d),
            MHCAFormer(patch_size=3, dim=dims[3], reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d)
        ])
        from Convnextv2.MLPDecoder import PLPDecoder
        self.decoder = PLPDecoder(in_channels=[96, 192, 384, 768], num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                   embed_dim=768)

    def forward(self, rgb, thermal):
        raw_rgb = rgb
        token_corrector = []
        for i in range(self.num_stages):
            cat = torch.cat([raw_rgb, thermal], dim=1)
            token = self.Token_corrector[i](cat)
            token_corrector.append(token)

        enc_rgb = self.encoder_rgb(rgb)
        enc_thermal = self.encoder_thermal(thermal)
        enc_feats = []
        for i in range(self.num_stages):
            vi, ir = self.csf[i](enc_rgb[i], enc_thermal[i])
            x_fused = self.crossFormer[i](vi, ir, token_corrector[i])
            enc_feats.append(x_fused)

        dec_out = self.decoder(enc_feats)
        output = F.interpolate(dec_out, size=raw_rgb.size()[-2:], mode='bilinear',
                               align_corners=True)
        return output

