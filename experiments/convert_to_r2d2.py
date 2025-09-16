import h5py
import os
import json

# Paths
src_dir = os.path.expanduser("~/xtrainer/experiments/datasets/Pick_Place/train_data")
dst_dir = os.path.expanduser("~/xtrainer/experiments/datasets/Pick_Place/train_data_r2d2")
os.makedirs(dst_dir, exist_ok=True)

# Number of episodes
num_episodes = 50

# Convert each episode to R2D2 format
for i in range(num_episodes):
    src_file = os.path.join(src_dir, f"episode_init_{i}.hdf5")
    dst_file = os.path.join(dst_dir, f"episode_init_{i}_r2d2.hdf5")
    
    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:
        # Create top-level 'data' group
        data_grp = dst.create_group("data")
        ep_grp = data_grp.create_group(str(i))
        
        # Copy top-level keys from src into episode group
        for key in src.keys():
            src.copy(key, ep_grp)
        
        # Optional: copy environment metadata if it exists
        if "env_args" in src.attrs:
            ep_grp.attrs["env_args"] = src.attrs["env_args"]

print(f"✅ Conversion complete. All {num_episodes} episodes are now in R2D2 format at:\n{dst_dir}")

# Generate the config data list for your experiment
config_data_list = [
    {"path": os.path.join(dst_dir, f"episode_init_{i}_r2d2.hdf5")} for i in range(num_episodes)
]

# Save as JSON for direct inclusion in your config
with open(os.path.join(dst_dir, "r2d2_data_paths.json"), "w") as f:
    json.dump(config_data_list, f, indent=4)

print(f"✅ Config paths JSON saved at: {os.path.join(dst_dir, 'r2d2_data_paths.json')}")
