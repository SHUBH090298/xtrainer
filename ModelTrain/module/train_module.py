import os
import torch
import numpy as np
import pickle
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import time
from torchvision import transforms
import torch.nn.functional as F

from ModelTrain.module.utils import (
    load_data,
    compute_dict_mean,
    set_seed,
    detach_dict,
    calibrate_linear_vel,
    postprocess_base_action,
)
from ModelTrain.module.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

import IPython

# Expose an embed function for debugging
e = IPython.embed


def _arg(args, key, default=None):
    """Helper: accept either dict-like args or argparse.Namespace-like args."""
    try:
        if isinstance(args, dict):
            return args.get(key, default)
        else:
            return getattr(args, key, default)
    except Exception:
        return default


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def train(args):
    seed = _arg(args, "seed", 1)
    set_seed(seed)

    ckpt_dir = _arg(args, "ckpt_dir")
    policy_class = "ACT"
    task_name = _arg(args, "task_name")
    batch_size_train = _arg(args, "batch_size")
    batch_size_val = _arg(args, "batch_size")
    num_steps = _arg(args, "num_steps")
    eval_every = _arg(args, "eval_every")
    validate_every = _arg(args, "validate_every")
    save_every = _arg(args, "save_every")
    resume_ckpt_path = _arg(args, "resume_ckpt_path")

    from ModelTrain.constants import TASK_CONFIGS

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

    # Policy hyperparameters (kept from original behavior)
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': _arg(args, 'lr'),
            'num_queries': _arg(args, 'chunk_size'),
            'kl_weight': _arg(args, 'kl_weight'),
            'hidden_dim': _arg(args, 'hidden_dim'),
            'dim_feedforward': _arg(args, 'dim_feedforward'),
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'vq': False,
            'vq_class': None,
            'vq_dim': None,
            'action_dim': 16,
            'no_encoder': _arg(args, 'no_encoder', False),
        }
    elif policy_class == "Diffusion":
        policy_config = {
            'lr': _arg(args, 'lr'),
            'camera_names': camera_names,
            'action_dim': 16,
            'observation_horizon': 1,
            'action_horizon': 8,
            'prediction_horizon': _arg(args, 'chunk_size'),
            'num_queries': _arg(args, 'chunk_size'),
            'num_inference_timesteps': 10,
            'ema_power': 0.75,
            'vq': False,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            'lr': _arg(args, 'lr'),
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
        }
    else:
        raise NotImplementedError(f"Unknown policy class: {policy_class}")

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': _arg(args, 'lr'),
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'temporal_agg': _arg(args, 'temporal_agg'),
        'camera_names': camera_names,
        'load_pretrain': _arg(args, 'load_pretrain', False),
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    expr_name = os.path.basename(ckpt_dir)

    # Load data
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        batch_size_train,
        batch_size_val,
        _arg(args, "chunk_size"),
        _arg(args, "skip_mirrored_data"),
        config["load_pretrain"],
        policy_class,
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
    )

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

    # Ensure we have a valid best checkpoint info
    if best_ckpt_info is None:
        print("No best checkpoint found during training. Saving last model instead.")
        torch.save(policy.serialize(), os.path.join(ckpt_dir, "policy_last.ckpt"))
        return

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step {best_step}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    # Delegate to policy class to configure optimizer
    return policy.configure_optimizers()


def get_image(ts, camera_names, rand_crop_resize=False):
    """Build a batched tensor of images for the given camera names.

    Returns a tensor of shape (1, num_cameras, C, H, W) on CUDA.
    """
    curr_images = []
    for cam_name in camera_names:
        img = ts.observation["images"][cam_name]  # expected H, W, C (uint8)
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                # convert to C, H, W
                img_chw = np.transpose(img, (2, 0, 1))
            elif img.ndim == 2:
                # single channel
                img_chw = np.expand_dims(img, 0)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            curr_images.append(img_chw)
        else:
            raise TypeError("Image must be a numpy array")

    curr_image = np.stack(curr_images, axis=0)  # num_cams, C, H, W
    curr_image = torch.from_numpy(curr_image).float() / 255.0  # scale

    # Optionally do a (deterministic) center crop + resize back to original size
    if rand_crop_resize:
        num_cams, C, H, W = curr_image.shape
        ratio = 0.95
        crop_h = max(1, int(H * ratio))
        crop_w = max(1, int(W * ratio))
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        curr_image = curr_image[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        # resize back
        curr_image = F.interpolate(curr_image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)

    # move to device and add batch dim
    curr_image = curr_image.cuda().unsqueeze(0)
    return curr_image


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config.get("seed", 0)
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    eval_every = config["eval_every"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]

    set_seed(seed)
    policy = make_policy(policy_class, policy_config)

    if config.get("load_pretrain", False):
        pretrain_path = os.path.join("/home/interbotix_ws/src/act/ckpts/pretrain_all", "policy_step_50000_seed_0.ckpt")
        if os.path.exists(pretrain_path):
            loading_status = policy.deserialize(torch.load(pretrain_path))
            print(f"loaded! {loading_status}")
        else:
            print("Pretrain path does not exist, skipping load_pretrain.")

    if config.get("resume_ckpt_path") is not None:
        if os.path.exists(config["resume_ckpt_path"]):
            loading_status = policy.deserialize(torch.load(config["resume_ckpt_path"]))
            print(f"Resume policy from: {config['resume_ckpt_path']}, Status: {loading_status}")
        else:
            print(f"resume_ckpt_path {config['resume_ckpt_path']} does not exist, skipping.")

    optimizer = make_optimizer(policy_class, policy)
    policy.cuda()

    min_val_loss = np.inf
    best_ckpt_info = None

    train_iter = repeater(train_dataloader)
    train_loss = []
    val_loss = []
    start_time = time.time()

    for step in tqdm(range(num_steps + 1)):
        # Validation
        if validate_every > 0 and step % validate_every == 0:
            print("validating")
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)

                    # Ensure forward_dict is a dict
                    if not isinstance(forward_dict, dict):
                        raise TypeError(f"policy returned {type(forward_dict)} from forward pass; expected dict")

                    # Guard against unhashable keys (e.g. lists) coming from policy
                    bad_keys = [k for k in forward_dict.keys() if not isinstance(k, (str, int, tuple))]
                    if bad_keys:
                        print("Warning: unhashable/nonstandard keys in forward_dict. Converting keys to strings:", bad_keys)
                        forward_dict = {str(k): v for k, v in forward_dict.items()}

                    validation_dicts.append(forward_dict)

                    # limit validation batches
                    if batch_idx >= 50:
                        break

                if len(validation_dicts) == 0:
                    print("No validation data collected (empty val_dataloader?).")
                else:
                    validation_summary = compute_dict_mean(validation_dicts)

                    if "loss" not in validation_summary:
                        raise KeyError("validation_summary missing 'loss' key. Got keys: {}".format(list(validation_summary.keys())))

                    epoch_val_loss = validation_summary["loss"].mean()

                    if epoch_val_loss < min_val_loss:
                        min_val_loss = float(epoch_val_loss)
                        best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))

                    # prefix keys with val_
                    val_prefixed = {f"val_{k}": v for k, v in validation_summary.items()}

                    print(f"Val loss:   {epoch_val_loss:.5f}")
                    try:
                        val_loss.append(float(epoch_val_loss.item()))
                    except Exception:
                        val_loss.append(float(epoch_val_loss))

                    summary_string = ""
                    for k, v in val_prefixed.items():
                        if hasattr(v, "mean"):
                            mean_val = v.mean().item()
                        else:
                            try:
                                mean_val = float(np.mean(v))
                            except Exception:
                                mean_val = float(v)
                        summary_string += f"{k}: {mean_val:.3f} "
                    print(summary_string)

        # Training step
        if step > 0 and eval_every > 0 and step % eval_every == 0:
            ckpt_name = f"policy_step_{step}_seed_{seed}.ckpt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        policy.train()
        optimizer.zero_grad()

        data = next(train_iter)
        forward_dict = forward_pass(data, policy)
        if not isinstance(forward_dict, dict):
            raise TypeError("policy forward must return a dict containing at least 'loss'")

        loss = forward_dict.get("loss")
        if loss is None:
            raise KeyError("forward_dict missing 'loss' key")

        loss.mean().backward()
        optimizer.step()

        try:
            train_loss.append(float(loss.mean().item()))
        except Exception:
            try:
                train_loss.append(float(loss.mean()))
            except Exception:
                train_loss.append(float(loss))

        # Periodic save
        if save_every > 0 and step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), ckpt_path)

    # End training loop
    total_time = time.time() - start_time
    print("train all time:", total_time)

    # Save last checkpoint
    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.serialize(), ckpt_path)

    # Ensure there's a best checkpoint
    if best_ckpt_info is None:
        print("No validation-improving checkpoint found; returning last model as best.")
        best_ckpt_info = (0, np.inf, deepcopy(policy.serialize()))

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_step_{best_step}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)

    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}")

    # Plot losses if we collected them
    if len(train_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Training Loss")
        plt.title("Training Loss Over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, "train_loss.png"))
        plt.close()

    if len(val_loss) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(val_loss, label="val Loss")
        plt.title("val Loss Over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, "val_loss.png"))
        plt.close()

    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f"Epoch {epoch} done")
        epoch += 1
