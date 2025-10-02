import torch
import torch.nn as nn

from ..nn.convs import Conv
from ..nn.blocks import SPPF, C2PSA, TFAttnBlock
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss
from .Head.detect import Detect


class TF_Attn_Yolo(BaseModel):
    def __init__(self, output_dir, num_classes=80, strides=[8, 16, 32], reg_max=16, device="cuda:0", input_canals=1, width_mult=0.25):
        super().__init__(device=device, output_dir=output_dir)
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max

        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)

        self.conv1 = Conv(input_canals, c1, k=3, s=2)
        self.conv2 = Conv(c1, c2, k=3, s=2)

        self.c3_1_in = Conv(c2, c3, k=1, s=1)
        self.c3_1 = TFAttnBlock(ch=c3, n=1, residual=True)

        self.conv3 = Conv(c3, c3, k=3, s=2)
        self.c3_2 = TFAttnBlock(ch=c3, n=1, residual=True)

        self.conv4 = Conv(c3, c4, k=3, s=2)
        self.c3_3 = TFAttnBlock(ch=c4, n=1, residual=True)

        self.conv5 = Conv(c4, c5, k=3, s=2)
        self.c3_4 = TFAttnBlock(ch=c5, n=1, residual=True)

        self.sppf = SPPF(c5, c5)
        self.attn = C2PSA(c1=c5, c2=c5, n=2, e=0.5)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.head_c3_1_in = Conv(c5 + c4, c4, k=1, s=1)
        self.head_c3_1 = TFAttnBlock(ch=c4, n=1, residual=True)

        self.head_c3_2_in = Conv(c4 + c3, c3, k=1, s=1)
        self.head_c3_2 = TFAttnBlock(ch=c3, n=1, residual=True)

        self.down_p3 = Conv(c3, c3, k=3, s=2)

        self.head_c3_3_in = Conv(c3 + c4, c4, k=1, s=1)
        self.head_c3_3 = TFAttnBlock(ch=c4, n=1, residual=True)

        self.down_p4 = Conv(c4, c4, k=3, s=2)

        self.head_c3_4_in = Conv(c4 + c5, c5, k=1, s=1)
        self.head_c3_4 = TFAttnBlock(ch=c5, n=1, residual=True)

        self.detect = Detect(
            in_channels=[c3, c4, c5],
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.strides
        )
        self.detect.bias_init(image_size=1024)

        self.criterion = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            device=self.device,
        )

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.c3_1_in(x)
        f2 = self.c3_1(x)

        x = self.conv3(f2)
        f3 = self.c3_2(x)

        x = self.conv4(f3)
        f4 = self.c3_3(x)

        x = self.conv5(f4)
        x = self.c3_4(x)
        x = self.sppf(x)
        f5 = self.attn(x)

        p5_up = self.upsample(f5)
        p4_feat = torch.cat([p5_up, f4], dim=1)
        p4_feat = self.head_c3_1_in(p4_feat)
        p4_out = self.head_c3_1(p4_feat)

        p4_up = self.upsample(p4_out)
        p3_feat = torch.cat([p4_up, f3], dim=1)
        p3_feat = self.head_c3_2_in(p3_feat)
        p3_out = self.head_c3_2(p3_feat)

        p3_down = self.down_p3(p3_out)
        pm_feat = torch.cat([p3_down, p4_out], dim=1)
        pm_feat = self.head_c3_3_in(pm_feat)
        p4_out2 = self.head_c3_3(pm_feat)

        p4_down = self.down_p4(p4_out2)
        pl_feat = torch.cat([p4_down, f5], dim=1)
        pl_feat = self.head_c3_4_in(pl_feat)
        p5_out = self.head_c3_4(pl_feat)

        outputs = self.detect(p3_out, p4_out2, p5_out)
        return outputs
