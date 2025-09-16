import h5py

file_path = "datasets/Pick_Place/train_data/episode_init_16.hdf5"

with h5py.File(file_path, "r") as f:
    def print_h5_keys(name, obj):
        print(f"{name} ({'Group' if isinstance(obj, h5py.Group) else 'Dataset'})")
    
    f.visititems(print_h5_keys)

    # Optionally, list top-level keys only
    print("\nTop-level keys:")
    for key in f.keys():
        print(key)
