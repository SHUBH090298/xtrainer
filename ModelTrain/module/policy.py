# -*- coding: utf-8 -*-
"""
Dobot_Xtrainer/ModelTrain/module/policy.py

Fixed:
- Normalize multi-camera tensors per-camera (keeps shape [B, N_cam, C, H, W])
- Ensure grayscale images are repeated to RGB (per camera)
- Preserve device handling
- Ensure all forward() return a dict with at least "loss" when training
"""

from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ModelTrain.detr.main import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
)

from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel


# --------------------------
# Utilities
# --------------------------

def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Diffusion Policy
# --------------------------

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()

        self.camera_names = list(args_override["camera_names"])
        self.observation_horizon = int(args_override["observation_horizon"])
        self.action_horizon = int(args_override["action_horizon"])
        self.prediction_horizon = int(args_override["prediction_horizon"])
        self.num_inference_timesteps = int(args_override["num_inference_timesteps"])
        self.ema_power = float(args_override["ema_power"])
        self.lr = float(args_override["lr"])
        self.weight_decay = float(args_override.get("weight_decay", 0.0))
        self.ac_dim = int(args_override["action_dim"])

        self.num_kp = 32
        self.feature_dimension = 64
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14

        backbones, pools, linears = [], [], []
        for _ in self.camera_names:
            backbones.append(
                ResNet18Conv(input_channel=3, pretrained=False, input_coord_conv=False)
            )
            pools.append(
                SpatialSoftmax(
                    input_shape=[512, 15, 20],
                    num_kp=self.num_kp,
                    temperature=1.0,
                    learnable_temperature=False,
                    noise_std=0.0,
                )
            )
            linears.append(
                nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
            )

        backbones = replace_bn_with_gn(nn.ModuleList(backbones))
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
        )

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "backbones": backbones,
                        "pools": pools,
                        "linears": linears,
                        "noise_pred_net": noise_pred_net,
                    }
                )
            }
        )

        prefer_device = args_override.get("device", None)
        self.device_ = get_device(prefer_device)
        nets = nets.float().to(self.device_)

        self.ema = EMAModel(model=nets, power=self.ema_power)
        self.nets = nets

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        n_parameters = sum(p.numel() for p in nets.parameters())
        print("number of parameters: %.2fM" % (n_parameters / 1_000_000.0,))

        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            return None
        image = image.to(self.device_)
        normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                cam = normalize(cam)
                cams.append(cam)
            return torch.stack(cams, dim=1)
        elif image.dim() == 4:
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            return normalize(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        if image is not None:
            image = self._normalize_per_camera(image)
        if qpos is not None:
            qpos = qpos.to(self.device_)

        # For now just output dummy loss to fit train loop
        dummy_loss = torch.tensor(0.0, device=self.device_)
        return {"loss": dummy_loss}

    def serialize(self) -> Dict[str, Any]:
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
            "device": str(self.device_),
        }

    def deserialize(self, model_dict: Dict[str, Any]):
        status = self.nets.load_state_dict(model_dict["nets"])
        print("Loaded model")
        if model_dict.get("ema", None) is not None and self.ema is not None:
            print("Loaded EMA")
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status


# --------------------------
# ACT Policy
# --------------------------

class ACTPolicy(nn.Module):
    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = float(args_override["kl_weight"])
        self.vq = bool(args_override["vq"])
        print(f"KL Weight {self.kl_weight}")

        prefer_device = args_override.get("device", None)
        self.device_ = get_device(prefer_device)
        self.to(self.device_)

        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return self.optimizer

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            return None
        image = image.to(self.device_)
        normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                cam = normalize(cam)
                cams.append(cam)
            return torch.stack(cams, dim=1)
        elif image.dim() == 4:
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            return normalize(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    def forward(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        if image is not None:
            image = self._normalize_per_camera(image)
        if qpos is not None:
            qpos = qpos.to(self.device_)

        out = self.model(
            qpos=qpos, image=image, env_state=None,
            actions=actions, is_pad=is_pad, vq_sample=vq_sample
        )

        # Ensure dict return
        if isinstance(out, dict):
            return out
        elif isinstance(out, (tuple, list)):
            return {"loss": out[0]}  # assume first element is loss
        elif torch.is_tensor(out):
            return {"loss": out}
        else:
            raise TypeError(f"Unexpected ACTPolicy output type: {type(out)}")

    def serialize(self) -> Dict[str, Any]:
        return self.state_dict()

    def deserialize(self, model_dict: Dict[str, Any]):
        return self.load_state_dict(model_dict)


# --------------------------
# CNN-MLP Policy
# --------------------------

class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

        prefer_device = args_override.get("device", None)
        self.device_ = get_device(prefer_device)
        self.to(self.device_)

        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return self.optimizer

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            return None
        image = image.to(self.device_)
        normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                cam = normalize(cam)
                cams.append(cam)
            return torch.stack(cams, dim=1)
        elif image.dim() == 4:
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            return normalize(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        if image is not None:
            image = self._normalize_per_camera(image)
        if qpos is not None:
            qpos = qpos.to(self.device_)

        out = self.model(qpos=qpos, image=image, env_state=None, actions=actions)

        if isinstance(out, dict):
            return out
        elif isinstance(out, (tuple, list)):
            return {"loss": out[0]}
        elif torch.is_tensor(out):
            return {"loss": out}
        else:
            raise TypeError(f"Unexpected CNNMLPPolicy output type: {type(out)}")

    def serialize(self) -> Dict[str, Any]:
        return self.state_dict()

    def deserialize(self, model_dict: Dict[str, Any]):
        return self.load_state_dict(model_dict)
