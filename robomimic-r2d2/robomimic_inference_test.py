import robomimic.utils.file_utils as FileUtils    
import robomimic.utils.torch_utils as TorchUtils    
import numpy as np    
import h5py    
import torch  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
from sklearn.metrics import mean_squared_error  
import os  
  
def plot_action_comparison(actual_actions, predicted_actions, action_names=None, save_path=None):  
    """  
    Plot comparison between actual and predicted actions  
    actual_actions: (T, D) array of ground truth actions  
    predicted_actions: (T, D) array of predicted actions  
    """  
    if action_names is None:  
        action_names = [f"Action_{i+1}" for i in range(actual_actions.shape[1] if len(actual_actions.shape) > 1 else 1)]  
      
    action_dim = actual_actions.shape[1] if len(actual_actions.shape) > 1 else 1  
    traj_length = len(actual_actions)  
      
    # Create subplots for each action dimension  
    fig, axs = plt.subplots(action_dim, 1, figsize=(15, action_dim * 3))  
    if action_dim == 1:  
        axs = [axs]  # Make it iterable for single action  
      
    for dim in range(action_dim):  
        ax = axs[dim]  
        if action_dim == 1:  
            actual_dim = actual_actions  
            predicted_dim = predicted_actions  
        else:  
            actual_dim = actual_actions[:, dim]  
            predicted_dim = predicted_actions[:, dim]  
              
        ax.plot(range(traj_length), actual_dim, label='Ground Truth', color='blue', linewidth=2)  
        ax.plot(range(traj_length), predicted_dim, label='Predicted', color='red', linewidth=2, linestyle='--')  
        ax.set_xlabel('Timestep')  
        ax.set_ylabel('Action Value')  
        ax.set_title(f'{action_names[dim] if dim < len(action_names) else f"Action_{dim+1}"}')  
        ax.legend()  
        ax.grid(True, alpha=0.3)  
      
    plt.tight_layout()  
      
    if save_path:  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
        print(f"Plot saved to: {save_path}")  
      
    plt.show()  
    return fig  
  
# Load the trained policy from checkpoint    
device = TorchUtils.get_torch_device(try_to_use_cuda=True)    
ckpt_path = "/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250916113358/models/model_epoch_50.pth"    
    
# Load policy using robomimic's utility    
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)    
    
# Check what observation keys the model expects    
shape_meta = ckpt_dict["shape_metadata"]    
expected_obs_keys = shape_meta["all_obs_keys"]    
print("Model expects these observation keys:", expected_obs_keys)    
    
# Path to your test episode data    
dataset_path = "/home/shubh/xtrainer/experiments/datasets/Pick_Place/merged_train_data_robomimic4.hdf5"    
  
# Add these variables before your demo loop  
all_predicted_actions = []  
all_ground_truth_actions = []  
demo_names = []  
  
# Run inference on test data    
with h5py.File(dataset_path, 'r') as root:    
    # Get list of demonstrations    
    demos = sorted([k for k in root["data"].keys() if k.startswith("demo_")])    
    print(f"Found {len(demos)} demonstrations")    
        
    # Test on first few demos    
    for demo_idx, demo_key in enumerate(demos[:3]):  # Test first 3 demos    
        print(f"\n=== Testing {demo_key} ===")    
            
        demo_group = root[f"data/{demo_key}"]    
        num_samples = demo_group.attrs["num_samples"]    
        print(f"Demo has {num_samples} samples")    
          
        demo_predicted = []  
        demo_ground_truth = []  
            
        # Test first few frames of this demo    
        for i in range(min(num_samples, 10)):  # Test more frames for better plots  
            obs_dict = {}    
                
            # Load low-dim observations - ensure they're numpy arrays with float64    
            if "obs/qpos" in demo_group and "qpos" in expected_obs_keys:    
                qpos_data = demo_group["obs/qpos"][i]    
                # Keep as numpy array but ensure float64 dtype    
                if qpos_data.shape == (1,):    
                    qpos_data = np.array([qpos_data[0]], dtype=np.float64)    
                else:    
                    qpos_data = qpos_data.astype(np.float64)    
                obs_dict["qpos"] = qpos_data    
                    
            if "obs/qvel" in demo_group and "qvel" in expected_obs_keys:    
                qvel_data = demo_group["obs/qvel"][i]    
                # Keep as numpy array but ensure float64 dtype    
                if qvel_data.shape == (1,):    
                    qvel_data = np.array([qvel_data[0]], dtype=np.float64)    
                else:    
                    qvel_data = qvel_data.astype(np.float64)    
                obs_dict["qvel"] = qvel_data    
                
            # Load RGB observations and convert to proper format for VisualCore    
            image_keys = {    
                "rightImg": "obs/rightImg",    
                "topImg": "obs/topImg",     
                "leftImg": "obs/leftImg"    
            }    
                
            for config_key, dataset_key in image_keys.items():    
                if dataset_key in demo_group and config_key in expected_obs_keys:    
                    # Load image data - shape: (480, 640, 3)    
                    img_data = demo_group[dataset_key][i]    
                        
                    # Convert to float32 and normalize to [0, 1]    
                    img_data = img_data.astype(np.float32) / 255.0    
                        
                    # Convert from (H, W, C) to (C, H, W) format as expected by VisualCore    
                    img_data = np.transpose(img_data, (2, 0, 1))    
                        
                    # Convert to numpy array first, then to tensor with single batch dimension    
                    obs_dict[config_key] = img_data  # Keep as numpy array    
                
            print(f"\nFrame {i}:")    
            print(f"Observation keys provided: {list(obs_dict.keys())}")    
                
            # Get action prediction - let robomimic handle tensor conversion    
            try:    
                with torch.no_grad():    
                    action = policy(obs_dict)    
                        
                print(f"Predicted action: {action}")    
                print(f"Action shape: {action.shape if hasattr(action, 'shape') else type(action)}")    
                    
                # Compare with ground truth if available    
                if "actions" in demo_group:    
                    gt_action = demo_group["actions"][i]    
                    print(f"Ground truth action: {gt_action}")    
                        
                    # Convert action to numpy for comparison    
                    if torch.is_tensor(action):    
                        action_np = action.detach().cpu().numpy().squeeze()    
                    else:    
                        action_np = np.array(action).squeeze()    
                        
                    if hasattr(gt_action, 'shape') and gt_action.shape == (1,):    
                        gt_action = gt_action[0]    
                      
                    # Collect for plotting  
                    demo_predicted.append(action_np)  
                    demo_ground_truth.append(gt_action)  
                        
                    if action_np.shape == gt_action.shape:    
                        action_delta = action_np - gt_action    
                        print(f"Action delta: {action_delta}")    
                        print(f"MSE: {np.mean(action_delta ** 2)}")    
                        print(f"Absolute error: {np.abs(action_delta)}")    
                    else:    
                        print(f"Shape mismatch: predicted {action_np.shape} vs ground truth {gt_action.shape}")    
                            
            except Exception as e:    
                print(f"Error during inference: {e}")    
                print(f"Available obs keys: {list(obs_dict.keys())}")    
                print(f"Expected obs keys: {expected_obs_keys}")    
                import traceback    
                traceback.print_exc()    
                break    
          
        # Store demo data for plotting  
        if demo_predicted and demo_ground_truth:  
            all_predicted_actions.append(np.array(demo_predicted))  
            all_ground_truth_actions.append(np.array(demo_ground_truth))  
            demo_names.append(demo_key)  
  
# After all demos, create plots  
print("\nGenerating action comparison plots...")  
for i, (pred, gt, name) in enumerate(zip(all_predicted_actions, all_ground_truth_actions, demo_names)):  
    save_path = f"action_comparison_{name}.png"  
    plot_action_comparison(gt, pred, action_names=[f"Action_1"], save_path=save_path)  
      
    # Calculate and print summary statistics  
    mse = mean_squared_error(gt, pred)  
    print(f"\n{name} Summary:")  
    print(f"  MSE: {mse:.2e}")  
    print(f"  Mean Absolute Error: {np.mean(np.abs(gt - pred)):.2e}")  
    print(f"  Max Absolute Error: {np.max(np.abs(gt - pred)):.2e}")  
  
print("Inference test completed!")