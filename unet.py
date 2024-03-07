from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
class Unet(nn.Module):

    def __init__(
            self,
            backbone='resnet50',
            freeze_backbone = False,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            num_classes=1,
            center=False,
            norm_layer=nn.BatchNorm2d,
            encoder_channels = None,
            use_img_space = False,
    ):
        super().__init__()
        
        encoder_channels = encoder_channels[::-1] # <--- reverse encoder channels
        self.encoder = backbone
        self.freeze_backbone = freeze_backbone 
        if self.freeze_backbone:
            for param in self.encoder.parameters():  
                param.requires_grad = False
        

        if not decoder_use_batchnorm:
            norm_layer = None

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            use_img_space=use_img_space,
        )
        print(f" The encoder channels are {encoder_channels[1:]}")
        print(f" The decoder channels are {decoder_channels}")
        print(f" The buttom layer is {encoder_channels[0]}")
        print(f" Using image space is {use_img_space}")
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x.reverse()  # reverse the order of the encoder feature maps
        x = self.decoder(x)
        return x


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,act_layer, norm_layer,
                  padding=0,stride=1, ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels,  **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            final_channels,
            norm_layer,
            center,
            use_img_space,
    ):
        super().__init__()
        ## POTENTIALLY CONVOLVE THE ENCODER OUTPUT ONTO ITSELF
        self.center = nn.Identity()
        if center:
            self.center = DecoderBlock(
                encoder_channels[0], 
                encoder_channels[0], 
                scale_factor=1.0, 
                norm_layer=norm_layer)
        # CALCULATE THE INPUT AND OUTPUT CHANNELS FOR EACH DECODER BLOCK
        skip_channels =     list(encoder_channels[1:]) + [0] if not use_img_space else encoder_channels[1:]
        upsample_channels = [encoder_channels[0]] + list(decoder_channels[:-1]) 
        in_channels = np.array(upsample_channels) + np.array(skip_channels)
        out_channels = decoder_channels
        # CREATE THE DECODER BLOCKS
        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(DecoderBlock(in_chs, out_chs,  norm_layer=norm_layer))
        
        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))
        self._init_weight()

        

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x

