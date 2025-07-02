import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class PLPDecoder(nn.Module):   # Progressive MLP decoder
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(PLPDecoder, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)

        self.linear_fuse43 = nn.Sequential(
            nn.Conv2d(in_channels=c3_in_channels + embedding_dim, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_fuse32 = nn.Sequential(
            nn.Conv2d(in_channels=c2_in_channels + embedding_dim, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_fuse21 = nn.Sequential(
            nn.Conv2d(in_channels=c1_in_channels + embedding_dim, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32

        c1, c2, c3, c4 = inputs[0],inputs[1],inputs[2],inputs[3]
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c3.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c43 = self.linear_fuse43(torch.cat([_c4, c3], dim=1))

        _c3 = self.linear_c3(c43).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c32 = self.linear_fuse32(torch.cat([_c3, c2], dim=1))

        _c2 = self.linear_c2(c32).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c21 = self.linear_fuse21(torch.cat([_c2, c1], dim=1))

        x = self.dropout(c21)
        x = self.linear_pred(x)

        return x

