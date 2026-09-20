# -*- coding: utf-8 -*-
"""切片分割 U-Net 模型定义（训练与推理共用，保证架构一致）。

支持两种编码器：
  - mobilenet_v2：torchvision 预训练编码器（Kaggle 上效果好，需联网下载权重）
  - none：从零训练的自定义轻量编码器（离线可用，无需预训练权重）

输出：5 类分割 logits，类别顺序见 SLICE_CLASSES：
  0=背景, 1=果肉, 2=果核, 3=褐变, 4=果皮
"""
import os
# Windows 下 torch+numpy(MKL) 可能发生 OpenMP 库冲突，提前规避（服务端推理同样适用）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn as nn

SLICE_CLASSES = ['背景', '果肉', '果核', '褐变', '果皮']
SLICE_CLASS_COLORS = [(0, 0, 0), (0, 255, 136), (0, 212, 255), (255, 71, 87), (255, 170, 60)]
N_CLASSES = len(SLICE_CLASSES)


def _conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _EncStage(nn.Module):
    """自定义编码器阶段：下采样 + 两次卷积"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            _conv_bn_relu(out_c, out_c),
            _conv_bn_relu(out_c, out_c),
        )

    def forward(self, x):
        return self.block(x)


class _DecStage(nn.Module):
    """解码器阶段：上采样 + 跳跃拼接 + 两次卷积"""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.conv = nn.Sequential(
            _conv_bn_relu(out_c + skip_c, out_c),
            _conv_bn_relu(out_c, out_c),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _mobilenet_encoder(pretrained):
    import torchvision.models as tv
    if pretrained:
        try:
            m = tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            m = tv.mobilenet_v2(weights=None)
    else:
        m = tv.mobilenet_v2(weights=None)
    return m.features


class UNet(nn.Module):
    """轻量 U-Net：输入 3 通道图，输出 n_classes 通道 logits（与原图同分辨率）。

    编码器三级下采样 + 跳跃连接：/2,/4,/8 或 /4,/8,/16（mobilenet_v2）。
    """

    def __init__(self, n_classes=N_CLASSES, base=32, backbone='none', pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = backbone
        self.base = base

        if backbone == 'mobilenet_v2':
            feats = _mobilenet_encoder(pretrained)
            # mobilenet_v2.features（torchvision 0.28 实测）：
            #   [0:3]=24@/4, [3:5]=32@/8, [5:8]=64@/16, [8:19]=1280@/32
            self.enc0 = feats[0:3]
            self.enc1 = feats[3:5]
            self.enc2 = feats[5:8]
            self.bottleneck = feats[8:19]
            enc_chs = [24, 32, 64, 1280]
            head_ups = 2          # enc0 步长 4 → 升采样 2 次回原分辨率
        else:
            self.enc0 = nn.Sequential(
                nn.Conv2d(3, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
                _conv_bn_relu(base, base),
            )
            self.enc1 = _EncStage(base, base * 2)
            self.enc2 = _EncStage(base * 2, base * 4)
            self.bottleneck = nn.Sequential(
                _EncStage(base * 4, base * 8),
                _conv_bn_relu(base * 8, base * 16),
                _conv_bn_relu(base * 16, base * 16),
            )
            enc_chs = [base, base * 2, base * 4, base * 16]
            head_ups = 1          # enc0 步长 2 → 升采样 1 次回原分辨率

        c0, c1, c2, c4 = enc_chs
        self.dec2 = _DecStage(c4, c2, c2)
        self.dec1 = _DecStage(c2, c1, c1)
        self.dec0 = _DecStage(c1, c0, c0)

        head = []
        ch = c0
        for _ in range(head_ups):
            head.append(nn.ConvTranspose2d(ch, max(8, ch // 2), 2, 2))
            head.append(nn.ReLU(inplace=True))
            ch = max(8, ch // 2)
        head.append(nn.Conv2d(ch, n_classes, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)


def build_unet(n_classes=N_CLASSES, base=32, backbone='none', pretrained=True):
    return UNet(n_classes=n_classes, base=base, backbone=backbone, pretrained=pretrained)


def save_checkpoint(model, path, extra=None):
    payload = {'state_dict': model.state_dict(),
               'config': {'n_classes': model.n_classes, 'base': model.base,
                          'backbone': model.backbone},
               'extra': extra or {}}
    torch.save(payload, str(path))
    return path


def load_checkpoint(path, device='cpu'):
    payload = torch.load(str(path), map_location=device, weights_only=False)
    cfg = payload.get('config', {})
    model = build_unet(n_classes=cfg.get('n_classes', N_CLASSES),
                       base=cfg.get('base', 32),
                       backbone=cfg.get('backbone', 'none'),
                       pretrained=False)
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()
    return model, payload