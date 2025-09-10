# -*- coding: utf-8 -*-
"""
Dobot_Xtrainer/ModelTrain/module/policy.py

Minimal edits to:
- Normalize multi-camera tensors per-camera (keeps shape [B, N_cam, C, H, W])
- Ensure grayscale images are repeated to RGB (per camera)
- Preserve device handling
"""

from typing import Dict, Any, Optional, Tuple

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
    """
    Choose a device safely. If 'prefer' is provided ('cuda' or 'cpu'), try that first.
    """
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Diffusion Policy
# --------------------------

class DiffusionPolicy(nn.Module):
    """
    Image-conditioned diffusion policy with per-camera ResNet18 backbones and SpatialSoftmax features.
    """

    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()

        # Required args
        self.camera_names = list(args_override["camera_names"])
        self.observation_horizon = int(args_override["observation_horizon"])
        self.action_horizon = int(args_override["action_horizon"])
        self.prediction_horizon = int(args_override["prediction_horizon"])
        self.num_inference_timesteps = int(args_override["num_inference_timesteps"])
        self.ema_power = float(args_override["ema_power"])
        self.lr = float(args_override["lr"])
        self.weight_decay = float(args_override.get("weight_decay", 0.0))
        self.ac_dim = int(args_override["action_dim"])

        # Model hyperparams (as in original)
        self.num_kp = 32
        self.feature_dimension = 64

        # Observation: concat features from all cams + qpos(=14 dims per original code)
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14

        # Build visual encoders (one per camera)
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(
                ResNet18Conv(
                    input_channel=3,
                    pretrained=False,
                    input_coord_conv=False,
                )
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

        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        # Replace BN with GN for stability
        backbones = replace_bn_with_gn(backbones)

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

        # Device handling
        prefer_device = args_override.get("device", None)
        self.device_ = get_device(prefer_device)
        nets = nets.float().to(self.device_)

        # EMA
        ENABLE_EMA = True
        self.ema = EMAModel(model=nets, power=self.ema_power) if ENABLE_EMA else None

        self.nets = nets

        # DDIM scheduler
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

        # base mean/std for a single camera (3 channels)
        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _encode_images(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-camera images into features.

        Expected image shape: [B, N_cam, C, H, W] where C=3.
        If C=1 (grayscale), will be repeated to RGB.
        Returns: concatenated features [B, feature_dimension * N_cam]
        """
        B, N_cam, C, H, W = image.shape
        assert N_cam == len(self.camera_names), "Mismatch in number of cameras"

        all_features = []
        backbones = self.nets["policy"]["backbones"]
        pools = self.nets["policy"]["pools"]
        linears = self.nets["policy"]["linears"]

        for cam_id in range(N_cam):
            cam_image = image[:, cam_id]  # [B, C, H, W]

            # Ensure 3-channel input
            if cam_image.shape[1] == 1:
                cam_image = cam_image.repeat(1, 3, 1, 1)

            cam_features = backbones[cam_id](cam_image)              # [B, 512, 15, 20]
            pool_features = pools[cam_id](cam_features)              # [B, num_kp, 2]
            pool_features = torch.flatten(pool_features, start_dim=1)  # [B, num_kp*2]
            out_features = linears[cam_id](pool_features)            # [B, feature_dimension]
            all_features.append(out_features)

        return torch.cat(all_features, dim=1)  # [B, feature_dimension * N_cam]

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B, N_cam, C, H, W] or [B, C, H, W]
        returns: same shape, normalized (and repeated to RGB if needed)
        """
        if image is None:
            return None
        image = image.to(self.device_)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]  # [B, C, H, W]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
                cam = normalize(cam)
                cams.append(cam)
            out = torch.stack(cams, dim=1)  # [B, N_cam, 3, H, W]
            return out
        elif image.dim() == 4:
            cam = image
            if cam.shape[1] == 1:
                cam = cam.repeat(1, 3, 1, 1)
            normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
            return normalize(cam)
        else:
            raise ValueError("Unexpected image shape for normalization: {}".format(image.shape))

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ):
        # Normalize per-camera and keep shape [B, N_cam, C, H, W]
        if image is not None:
            image = self._normalize_per_camera(image)

        if qpos is not None:
            qpos = qpos.to(self.device_)

        # Now delegate to model. DETRVAE.forward expects image as [B, N_cam, C, H, W]
        # and an env_state kwarg (may be None)
        # For the diffusion policy, the 'noise_pred_net' usage is usually internal in training.
        # Here we simply call the policy nets if required by your inference workflow.
        # If your code expects something else, adapt accordingly.
        return self.nets  # placeholder: training/inference pipeline should call the right subcomponents

    def serialize(self) -> Dict[str, Any]:
        return {
            "nets": self.nets.state_dict(),
            "ema": (
                self.ema.averaged_model.state_dict() if self.ema is not None else None
            ),
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
    """
    Autoregressive Transformer (ACT) policy wrapper.
    """

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
        # move module to device
        self.to(self.device_)

        # base mean/std for a single camera (3 channels)
        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return self.optimizer

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B, N_cam, C, H, W] or [B, C, H, W]
        returns: same shape, normalized (and repeated to RGB if needed)
        """
        if image is None:
            return None
        image = image.to(self.device_)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
                cam = normalize(cam)
                cams.append(cam)
            out = torch.stack(cams, dim=1)
            return out
        elif image.dim() == 4:
            cam = image
            if cam.shape[1] == 1:
                cam = cam.repeat(1, 3, 1, 1)
            normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
            return normalize(cam)
        else:
            raise ValueError("Unexpected image shape for normalization: {}".format(image.shape))

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        vq_sample: Optional[torch.Tensor] = None,
    ):
        # Normalize per-camera and keep shape [B, N_cam, C, H, W]
        if image is not None:
            image = self._normalize_per_camera(image)

        if qpos is not None:
            qpos = qpos.to(self.device_)

        # the underlying model (built by build_ACT_model_and_optimizer) expects:
        # DETRVAE.forward(qpos, image, env_state, actions=None, is_pad=None, vq_sample=None)
        # call with env_state=None for inference
        return self.model(qpos=qpos, image=image, env_state=None, actions=actions, is_pad=is_pad, vq_sample=vq_sample)

    @torch.no_grad()
    def vq_encode(self, qpos: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor):
        ...

    def serialize(self) -> Dict[str, Any]:
        return self.state_dict()

    def deserialize(self, model_dict: Dict[str, Any]):
        return self.load_state_dict(model_dict)


# --------------------------
# CNN-MLP Policy
# --------------------------

class CNNMLPPolicy(nn.Module):
    """
    Simple visual encoder + MLP action regressor.
    """

    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

        prefer_device = args_override.get("device", None)
        self.device_ = get_device(prefer_device)
        self.to(self.device_)

        # base mean/std for one camera
        self.base_mean = [0.485, 0.456, 0.406]
        self.base_std = [0.229, 0.224, 0.225]

    def configure_optimizers(self):
        return self.optimizer

    def _normalize_per_camera(self, image: torch.Tensor) -> torch.Tensor:
        if image is None:
            return None
        image = image.to(self.device_)
        if image.dim() == 5:
            B, N_cam, C, H, W = image.shape
            cams = []
            for cam_id in range(N_cam):
                cam = image[:, cam_id]
                if cam.shape[1] == 1:
                    cam = cam.repeat(1, 3, 1, 1)
                normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
                cam = normalize(cam)
                cams.append(cam)
            out = torch.stack(cams, dim=1)
            return out
        elif image.dim() == 4:
            cam = image
            if cam.shape[1] == 1:
                cam = cam.repeat(1, 3, 1, 1)
            normalize = transforms.Normalize(mean=self.base_mean, std=self.base_std)
            return normalize(cam)
        else:
            raise ValueError("Unexpected image shape for normalization: {}".format(image.shape))

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ):
        if image is not None:
            image = self._normalize_per_camera(image)

        if qpos is not None:
            qpos = qpos.to(self.device_)

        # CNNMLP expects env_state in its forward signature, pass None for inference
        return self.model(qpos=qpos, image=image, env_state=None, actions=actions)
