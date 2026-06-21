# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_torch_modules():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mujoco-unet vision requires optional dependencies torch and torchvision. "
            "Install them only when using --vision-mode mujoco-unet."
        ) from exc
    return torch, nn, models


def _convrelu(nn, in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def _projection(nn, c, n, k):
    return nn.Sequential(*[_convrelu(nn, c, c, k, k // 2) for _ in range(n)])


def _build_resnet_unet(nn, models, n_class: int = 2, projection_n: int = 1, projection_k: int = 1):
    class ResNetUNet(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.base_model = models.resnet18(weights=None)
            except TypeError:
                self.base_model = models.resnet18(pretrained=False)
            self.base_layers = list(self.base_model.children())

            self.layer0 = nn.Sequential(*self.base_layers[:3])
            self.layer0_1x1 = _projection(nn, 64, projection_n, projection_k)
            self.layer1 = nn.Sequential(*self.base_layers[3:5])
            self.layer1_1x1 = _projection(nn, 64, projection_n, projection_k)
            self.layer2 = self.base_layers[5]
            self.layer2_1x1 = _projection(nn, 128, projection_n, projection_k)
            self.layer3 = self.base_layers[6]
            self.layer3_1x1 = _projection(nn, 256, projection_n, projection_k)
            self.layer4 = self.base_layers[7]
            self.layer4_1x1 = _projection(nn, 512, projection_n, projection_k)

            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv_up3 = _convrelu(nn, 256 + 512, 512, 3, 1)
            self.conv_up2 = _convrelu(nn, 128 + 512, 256, 3, 1)
            self.conv_up1 = _convrelu(nn, 64 + 256, 256, 3, 1)
            self.conv_up0 = _convrelu(nn, 64 + 256, 128, 3, 1)

            self.conv_original_size0 = _convrelu(nn, 3, 64, 3, 1)
            self.conv_original_size1 = _convrelu(nn, 64, 64, 3, 1)
            self.conv_original_size2 = _convrelu(nn, 64 + 128, 64, 3, 1)
            self.conv_last = nn.Conv2d(64, n_class, 1)

        def forward(self, input_tensor):
            x_original = self.conv_original_size0(input_tensor)
            x_original = self.conv_original_size1(x_original)

            layer0 = self.layer0(input_tensor)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            layer4 = self.layer4_1x1(layer4)
            x = self.upsample(layer4)
            layer3 = self.layer3_1x1(layer3)
            x = self.conv_up3(torch_cat([x, layer3], dim=1))

            x = self.upsample(x)
            layer2 = self.layer2_1x1(layer2)
            x = self.conv_up2(torch_cat([x, layer2], dim=1))

            x = self.upsample(x)
            layer1 = self.layer1_1x1(layer1)
            x = self.conv_up1(torch_cat([x, layer1], dim=1))

            x = self.upsample(x)
            layer0 = self.layer0_1x1(layer0)
            x = self.conv_up0(torch_cat([x, layer0], dim=1))

            x = self.upsample(x)
            x = self.conv_original_size2(torch_cat([x, x_original], dim=1))
            return self.conv_last(x)

    def torch_cat(xs, dim):
        # Captured as a tiny indirection so the nested model stays isolated from
        # module-level torch imports.
        return xs[0].new_empty(0) if False else __import__("torch").cat(xs, dim=dim)

    return ResNetUNet()


def infer_unet_heatmaps(image_rgb: np.ndarray, model_path: Path, device: str = "cpu") -> np.ndarray:
    """Run the two-keypoint U-Net used by the visual-servoing project.

    The external ROS annotator publishes two heatmap argmax points. This helper
    keeps that inference shape but avoids importing ROS or parsing CLI args.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"U-Net model file not found: {model_path}")
    torch, nn, models = _load_torch_modules()

    device_obj = torch.device(device)
    model = _build_resnet_unet(nn, models, n_class=2).to(device_obj)
    try:
        payload = torch.load(model_path, map_location=device_obj, weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location=device_obj)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    image = np.asarray(image_rgb, dtype=np.float32) / 255.0
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image shaped [H,W,3], got {image.shape}")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(device_obj)

    use_half = device_obj.type == "cuda"
    if use_half:
        model = model.half()
        tensor = tensor.half()
    with torch.no_grad():
        output = model(tensor).squeeze(0)
    return output.detach().float().cpu().numpy()
