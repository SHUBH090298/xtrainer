import h5py
import os
import numpy as np

input_dir = "/home/shubh/xtrainer/experiments/datasets/Pick_Place/train_data"
output_file = "/home/shubh/xtrainer/experiments/datasets/Pick_Place/train_data_merged.h5"

h5_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".h5")]
h5_files.sort()

def merge_datasets(in_h5, out_h5, path=""):
    for key in in_h5.keys():
        in_obj = in_h5[key]
        out_path = f"{path}/{key}" if path else key

        if isinstance(in_obj, h5py.Group):
            if out_path not in out_h5:
                out_h5.create_group(out_path)
            merge_datasets(in_obj, out_h5, out_path)

        elif isinstance(in_obj, h5py.Dataset):
            data = in_obj[:]
            if data.size == 0:
                print(f"⚠ Skipping empty dataset: {out_path}")
                continue

            # Add a batch dimension if necessary
            if out_path in out_h5:
                # Stack along axis 0
                existing = out_h5[out_path]
                new_shape = (existing.shape[0] + 1,) + existing.shape[1:]
                existing.resize(new_shape)
                existing[-1:] = np.expand_dims(data, axis=0)
                print(f"Appended dataset: {out_path}, shape {data.shape}")
            else:
                # Create with unlimited first dimension
                maxshape = (None,) + data.shape
                out_h5.create_dataset(out_path, data=np.expand_dims(data, axis=0), maxshape=maxshape, chunks=True)
                print(f"Created dataset: {out_path}, shape {data.shape}")

with h5py.File(output_file, 'w') as out_h5:
    for i, file_path in enumerate(h5_files):
        with h5py.File(file_path, 'r') as in_h5:
            print(f"Merging {file_path} ({i+1}/{len(h5_files)})")
            merge_datasets(in_h5, out_h5)

print("✅ All HDF5 files merged into:", output_file)

# Inspect merged file
with h5py.File(output_file, 'r') as f:
    def print_h5_keys(name, obj):
        print(f"{name}: {'Group' if isinstance(obj, h5py.Group) else 'Dataset'}, shape: {getattr(obj, 'shape', '-')}")
    f.visititems(print_h5_keys)
