import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelEmbed1(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed1, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):

        x = self.channel_embed(x)
        return x

class CSFNet(nn.Module):
    def __init__(self, channels):
        super(CSFNet, self).__init__()
        self.cs_afm = CosineSimilarityAttentionFusionModule(channels)

    def forward(self, rgb, thermal):
        Fx, Fy = self.cs_afm(rgb, thermal)
        return Fx, Fy

# 使用示例
class CosineSimilarityAttentionFusionModule(nn.Module):
    def __init__(self, channels):
        super(CosineSimilarityAttentionFusionModule, self).__init__()
    def forward(self, Fx, Fy):
        B, C, H, W = Fx.size()
        x_reshaped = Fx.view(B, C, -1)
        y_reshaped = Fy.view(B, C, -1)
        cosine_similarities = F.cosine_similarity(x_reshaped, y_reshaped, dim=2)
        cosine_similarities_expanded = cosine_similarities.unsqueeze(2)
        w = torch.relu(cosine_similarities_expanded)
        w = w.unsqueeze(-1)
        w_x = w * Fx + Fx
        w_y = (1-w) * Fy + Fy
        return w_x, w_y
