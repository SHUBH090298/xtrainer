# model_module.py
import os
import time
import pickle
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
import matplotlib.pyplot as plt

from module.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from detr.models.latent_model import Latent_Model_Transformer
from ModelTrain.model_train import arg_config


def set_config():
    args = arg_config()
    ckpt_dir = args["ckpt_dir"]
    policy_class = "ACT"
    task_name = args.get("task_name", None)
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]

    # Fix for decompiled bug: robustly check whether task is simulation
    is_sim = isinstance(task_name, str) and task_name.startswith("sim_")

    if is_sim or task_name == "all":
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config["dataset_dir"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)

    state_dim = 14
    lr_backbone = 1e-05
    backbone = "resnet18"

    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "vq": False,
            "vq_class": None,
            "vq_dim": None,
            "action_dim": 16,
            "no_encoder": args["no_encoder"],
        }
    else:
        if policy_class == "Diffusion":
            policy_config = {
                "lr": args["lr"],
                "camera_names": camera_names,
                "action_dim": 16,
                "observation_horizon": 1,
                "action_horizon": 8,
                "prediction_horizon": args["chunk_size"],
                "num_queries": args["chunk_size"],
                "num_inference_timesteps": 10,
                "ema_power": 0.75,
                "vq": False,
            }
        elif policy_class == "CNNMLP":
            policy_config = {
                "lr": args["lr"],
                "lr_backbone": lr_backbone,
                "backbone": backbone,
                "num_queries": 1,
                "camera_names": camera_names,
            }
        else:
            raise NotImplementedError

    config = {
        "num_steps": num_steps,
        "eval_every": eval_every,
        "validate_every": validate_every,
        "save_every": save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": resume_ckpt_path,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "load_pretrain": args["load_pretrain"],
    }
    return config


def _to_tensor_actions(x, device):
    """
    Converts list/tuple/np.array/torch.Tensor of actions to a stacked tensor on given device.
    Handles nested lists, numpy arrays, torch tensors, or tuple outputs.

    device may be a torch.device or a string like "cuda"/"cpu".
    """
    # unwrap tuple if needed
    if isinstance(x, tuple):
        x = x[0]

    # convert list of arrays/tensors to single tensor
    tensor_list = []
    for a in x:
        if isinstance(a, torch.Tensor):
            tensor_list.append(a.detach().cpu())
        elif isinstance(a, np.ndarray):
            tensor_list.append(torch.from_numpy(a).float())
        elif isinstance(a, list):
            tensor_list.append(torch.tensor(a, dtype=torch.float32))
        else:
            # try to coerce scalars
            try:
                tensor_list.append(torch.tensor(a, dtype=torch.float32))
            except Exception as e:
                raise TypeError(f"Unsupported type for action element: {type(a)} -- {e}")
    out = torch.stack(tensor_list).float()
    # accept device as string or torch.device
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    return out.to(device_obj)


def _extract_action_tensor(obj: Any, device: str = "cpu"):
    """
    Try to extract a meaningful (T x dim) tensor-like object from policy output.
    Supports: torch.Tensor, np.ndarray, list, tuple, dict (recursively).
    Returns: torch.Tensor on requested device.
    Raises TypeError if nothing suitable found.
    """
    # Direct tensor/ndarray/list/tuple
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float().to(device)
    if isinstance(obj, (list, tuple)):
        # if list of tensors/arrays -> stack
        # if list contains scalars -> convert
        return _to_tensor_actions(list(obj), device)
    if isinstance(obj, dict):
        # try common keys first
        for key in ("actions", "action", "pred", "output", "outputs", "outputs_pred"):
            if key in obj:
                try:
                    return _extract_action_tensor(obj[key], device)
                except Exception:
                    pass
        # otherwise iterate values and pick the first value that yields a tensor
        for v in obj.values():
            try:
                return _extract_action_tensor(v, device)
            except Exception:
                continue
        raise TypeError("Could not extract tensor from dict policy output.")
    # last attempt: try to convert to tensor
    try:
        return torch.tensor(obj, dtype=torch.float32).to(device)
    except Exception:
        raise TypeError(f"Unsupported action output type: {type(obj)}")


class Imitate_Model:
    def __init__(self, ckpt_dir=None, ckpt_name="policy_last.ckpt"):
        config = set_config()
        self.config = config
        self.ckpt_name = ckpt_name

        self.ckpt_dir = ckpt_dir if ckpt_dir else config["ckpt_dir"]
        self.state_dim = config["state_dim"]
        self.policy_class = config["policy_class"]
        self.policy_config = config["policy_config"]
        self.camera_names = config["camera_names"]
        self.max_timesteps = config["episode_len"]
        self.temporal_agg = config["temporal_agg"]
        self.vq = config["policy_config"].get("vq", False)
        self.t = 0
        self.latent_model = None
        self.vq_sample = None
        self.query_frequency = None
        self.num_queries = None
        self.all_actions = None
        self.all_time_actions = None

    def __make_policy(self):
        if self.policy_class == "ACT":
            return ACTPolicy(self.policy_config)
        elif self.policy_class == "CNNMLP":
            return CNNMLPPolicy(self.policy_config)
        elif self.policy_class == "Diffusion":
            return DiffusionPolicy(self.policy_config)
        else:
            raise NotImplementedError

    def __image_process(self, observation, camera_names, rand_crop_resize=False):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(observation["images"][cam_name], "h w c -> c h w")
            curr_images.append(curr_image)

        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float()

        num_cams, C, H, W = curr_image.shape
        curr_image = curr_image.unsqueeze(0)

        if rand_crop_resize:
            ratio = 0.95
            crop_h = int(H * ratio)
            crop_w = int(W * ratio)
            y0 = (H - crop_h) // 2
            x0 = (W - crop_w) // 2
            y1 = y0 + crop_h
            x1 = x0 + crop_w
            curr_image = curr_image[..., y0:y1, x0:x1].squeeze(0)
            resize_transform = transforms.Resize((H, W), antialias=True)
            resized = []
            for cam in curr_image:
                resized_cam = resize_transform(cam)
                resized.append(resized_cam)
            curr_image = torch.stack(resized, dim=0).unsqueeze(0)

        if torch.cuda.is_available():
            curr_image = curr_image.cuda()

        return curr_image

    def __get_auto_index(self, dataset_dir):
        max_idx = 1000
        for i in range(max_idx + 1):
            if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
                return i
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

    def loadModel(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.dirname(cur_path)
        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(dir_path, ckpt_path)

        self.policy = self.__make_policy()
        loaded = torch.load(ckpt_path, map_location="cpu")
        loading_status = self.policy.deserialize(loaded)
        print("policy deserialize status:", loading_status)
        if torch.cuda.is_available():
            self.policy.cuda()
        self.policy.eval()

        if self.vq:
            vq_dim = self.config["policy_config"]["vq_dim"]
            vq_class = self.config["policy_config"]["vq_class"]
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(self.ckpt_dir, "latent_model_last.ckpt")
            if not os.path.isabs(latent_model_ckpt_path):
                latent_model_ckpt_path = os.path.join(dir_path, latent_model_ckpt_path)
            latent_loaded = torch.load(latent_model_ckpt_path, map_location="cpu")
            latent_model.deserialize(latent_loaded)
            latent_model.eval()
            if torch.cuda.is_available():
                latent_model.cuda()
            self.latent_model = latent_model
            print(f"Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}")
        else:
            print(f"Loaded: {ckpt_path}")

        stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
        if not os.path.isabs(stats_path):
            stats_path = os.path.join(dir_path, stats_path)

        # load stats safely and provide fallbacks if action normalization missing
        try:
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load stats from {stats_path}: {e}")
            stats = {}

        # pre_process for qpos
        if "qpos_mean" in stats and "qpos_std" in stats:
            self.pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
        else:
            print("Warning: qpos mean/std not found in stats - using identity pre_process")
            self.pre_process = lambda s_qpos: s_qpos

        # post_process for actions (support both mean/std or min/max or fallback identity)
        if self.policy_class == "Diffusion":
            if "action_max" in stats and "action_min" in stats:
                self.post_process = lambda a: (a + 1) / 2 * (stats["action_max"] - stats["action_min"]) + stats["action_min"]
            else:
                print("Warning: action_min/action_max not found in stats - using identity post_process for Diffusion")
                self.post_process = lambda a: a
        else:
            if "action_std" in stats and "action_mean" in stats:
                self.post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
            else:
                print("Warning: action_mean/action_std not found in stats - using identity post_process")
                self.post_process = lambda a: a

        self.query_frequency = int(self.policy_config.get("num_queries", 1))
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = int(self.policy_config.get("num_queries", 1))
        else:
            self.num_queries = self.query_frequency

        self.max_timesteps = int(self.max_timesteps * 1)
        self.episode_returns = []
        self.highest_rewards = []

        if self.temporal_agg:
            buffer_len = self.max_timesteps + self.num_queries
            if torch.cuda.is_available():
                self.all_time_actions = torch.zeros([self.max_timesteps, buffer_len, 16]).cuda()
            else:
                self.all_time_actions = torch.zeros([self.max_timesteps, buffer_len, 16])

        self.qpos_history_raw = np.zeros((self.max_timesteps, self.state_dim))
        self.image_list = []
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []
        self.all_actions = None

    def predict(self, observation, t, save_qpos_history=False):
        with torch.inference_mode():
            qpos_numpy = np.array(observation["qpos"])
            if t < self.qpos_history_raw.shape[0]:
                self.qpos_history_raw[t] = qpos_numpy
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float()
            if torch.cuda.is_available():
                qpos = qpos.cuda()
            qpos = qpos.unsqueeze(0)

            curr_image = None
            if t % self.query_frequency == 0:
                curr_image = self.__image_process(observation, self.camera_names, rand_crop_resize=(self.policy_class == "Diffusion"))

            if t == 0:
                for _ in range(10):
                    try:
                        if curr_image is None:
                            curr_image = self.__image_process(observation, self.camera_names, rand_crop_resize=(self.policy_class == "Diffusion"))
                        _ = self.policy(qpos, curr_image)
                    except Exception:
                        pass
                print("network warm up done")

            # Main action computation
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

            if self.policy_class == "ACT":
                if t % self.query_frequency == 0 or self.all_actions is None:
                    if curr_image is None:
                        curr_image = self.__image_process(observation, self.camera_names)
                    if self.vq and self.latent_model is not None:
                        self.vq_sample = self.latent_model.generate(1, temperature=1, x=None)
                        self.all_actions = self.policy(qpos, curr_image, vq_sample=self.vq_sample)
                    else:
                        self.all_actions = self.policy(qpos, curr_image)

                all_actions_local = self.all_actions

                # if tuple -> take first element (common pattern)
                if isinstance(all_actions_local, tuple):
                    all_actions_local = all_actions_local[0]

                # If it's already tensor/ndarray/list -> extract/convert to tensor on device
                try:
                    # try extracting a tensor (this function handles dicts as well)
                    all_actions_local_tensor = _extract_action_tensor(all_actions_local, device_name)
                except TypeError:
                    # fallback: if it is list-like but not handled, try conversion
                    if isinstance(all_actions_local, (list, tuple, np.ndarray)):
                        all_actions_local_tensor = _to_tensor_actions(list(all_actions_local), device_name)
                    else:
                        raise

                # determine length safely
                if isinstance(all_actions_local_tensor, torch.Tensor):
                    length = all_actions_local_tensor.shape[0]
                else:
                    try:
                        length = len(all_actions_local_tensor)
                    except Exception:
                        length = 1

                qidx = min(int(t % self.query_frequency), max(0, int(length) - 1))

                # index into tensor
                if isinstance(all_actions_local_tensor, torch.Tensor):
                    raw_action_tensor = all_actions_local_tensor[qidx]
                    raw_action = raw_action_tensor.detach().cpu().numpy()
                else:
                    # convert and index
                    raw_action = np.array(all_actions_local_tensor[qidx])

            else:
                # Non-ACT policies: Diffusion or CNNMLP
                if t % self.query_frequency == 0 or self.all_actions is None:
                    if curr_image is None:
                        curr_image = self.__image_process(observation, self.camera_names)
                    self.all_actions = self.policy(qpos, curr_image)

                all_actions_local = self.all_actions
                if isinstance(all_actions_local, tuple):
                    all_actions_local = all_actions_local[0]

                # Attempt to extract tensor/array
                try:
                    all_actions_local_tensor = _extract_action_tensor(all_actions_local, device_name)
                except TypeError:
                    if isinstance(all_actions_local, (list, tuple, np.ndarray)):
                        all_actions_local_tensor = _to_tensor_actions(list(all_actions_local), device_name)
                    else:
                        raise

                if isinstance(all_actions_local_tensor, torch.Tensor):
                    length = all_actions_local_tensor.shape[0]
                else:
                    try:
                        length = len(all_actions_local_tensor)
                    except Exception:
                        length = 1

                qidx = min(int(t % self.query_frequency), max(0, int(length) - 1))

                if isinstance(all_actions_local_tensor, torch.Tensor):
                    raw_action_tensor = all_actions_local_tensor[qidx]
                    raw_action = raw_action_tensor.detach().cpu().numpy()
                else:
                    raw_action = np.array(all_actions_local_tensor[qidx])

            # final post-processing and splitting into qpos target and base action
            action = self.post_process(raw_action)

            if action.shape[0] >= 2:
                target_qpos = action[:-2]
                base_action = action[-2:]
            else:
                target_qpos = action
                base_action = np.zeros(2)

            self.qpos_list.append(qpos_numpy)
            self.target_qpos_list.append(target_qpos)

            if save_qpos_history:
                log_id = self.__get_auto_index(self.ckpt_dir)
                np.save(os.path.join(self.ckpt_dir, f"qpos_{log_id}.npy"), self.qpos_history_raw)
                plt.figure(figsize=(10, 20))
                for i in range(self.state_dim):
                    plt.subplot(self.state_dim, 1, i + 1)
                    plt.plot(self.qpos_history_raw[:, i])
                    if i != self.state_dim - 1:
                        plt.xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(self.ckpt_dir, f"qpos_{log_id}.png"))
                plt.close()

        return target_qpos
