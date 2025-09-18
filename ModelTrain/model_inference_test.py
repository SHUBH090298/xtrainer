from robomimic.utils.file_utils import policy_from_checkpoint  
import cv2  
import h5py  
import numpy as np  
import json  
  
def preprocess_image_for_robomimic(img, target_size=(84, 84)):  
    """  
    Preprocess images for robomimic BC model.  
    Converts from HWC to CHW format and normalizes to [0, 1].  
    """  
    # Resize image  
    img_resized = cv2.resize(img, target_size)  
    # Convert to float and normalize to [0, 1]  
    img_normalized = img_resized.astype(np.float32) / 255.0  
    # Convert from HWC to CHW format  
    img_chw = np.transpose(img_normalized, (2, 0, 1))  
    return img_chw  
  
if __name__ == '__main__':  
    # Load robomimic BC model  
    model, ckpt_dict = policy_from_checkpoint(  
        ckpt_path="/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250917115237/models/model_epoch_100.pth"  
    )  
      
    print("Model loaded successfully with built-in normalization")  
      
    # Parse config to understand expected image sizes  
    config_json = ckpt_dict.get("config", "{}")  
    config_dict = json.loads(config_json)  
    obs_config = config_dict.get("observation", {})  
      
    # Try to extract image dimensions from config  
    rgb_config = obs_config.get("encoder", {}).get("rgb", {})  
    print("RGB encoder config:", rgb_config)  
      
    # Check if there are specific image dimensions in the config  
    core_kwargs = rgb_config.get("core_kwargs", {})  
    print("Core kwargs:", core_kwargs)  
      
    # Try different common image sizes  
    possible_sizes = [(84, 84), (224, 224), (128, 128), (96, 96), (64, 64)]  
      
    dataset_path = "/home/shubh/xtrainer/experiments/datasets/Pick_Place/merged_train_data_robomimic4.hdf5"  
      
    with h5py.File(dataset_path, 'r') as root:  
        demo_key = "demo_0"  
        demo_data = root[f"data/{demo_key}"]  
          
        # Test with first frame  
        qpos = demo_data["obs/qpos"][0]  
        qvel = demo_data["obs/qvel"][0]   
        action = demo_data["actions"][0]  
          
        # Load images  
        left_img = demo_data["obs/leftImg"][0]  
        right_img = demo_data["obs/rightImg"][0]  
        top_img = demo_data["obs/topImg"][0]  
          
        # If images are compressed, decode them  
        if len(left_img.shape) == 1:  
            left_img = cv2.imdecode(left_img, cv2.IMREAD_COLOR)  
            right_img = cv2.imdecode(right_img, cv2.IMREAD_COLOR)  
            top_img = cv2.imdecode(top_img, cv2.IMREAD_COLOR)  
          
        print(f"Original image shapes: Left: {left_img.shape}, Right: {right_img.shape}, Top: {top_img.shape}")  
          
        # Try different image sizes until one works  
        for target_size in possible_sizes:  
            print(f"\nTrying image size: {target_size}")  
              
            try:  
                # Preprocess images with current size  
                left_processed = preprocess_image_for_robomimic(left_img, target_size)  
                right_processed = preprocess_image_for_robomimic(right_img, target_size)    
                top_processed = preprocess_image_for_robomimic(top_img, target_size)  
                  
                obs_dict = {  
                    'qpos': qpos,  
                    'qvel': qvel,  
                    'leftImg': left_processed,  
                    'rightImg': right_processed,  
                    'topImg': top_processed  
                }  
                  
                # Try prediction  
                predicted_action = model(obs_dict)  
                  
                print(f"SUCCESS! Image size {target_size} works!")  
                print("Ground truth action (deg):", [np.rad2deg(j) for j in action])  
                print("Predicted action (deg):", [np.rad2deg(j) for j in predicted_action])  
                print("Action delta (deg):", [np.rad2deg(j) for j in (predicted_action - action)])  
                  
                # Use this size for the rest of the analysis  
                working_size = target_size  
                break  
                  
            except Exception as e:  
                print(f"Failed with size {target_size}: {str(e)[:100]}...")  
                continue  
        else:  
            print("ERROR: None of the common image sizes worked!")  
            exit(1)  
          
        # Continue with successful size for more frames  
        print(f"\nContinuing analysis with working size: {working_size}")  
          
        for i in range(1, min(10, demo_data.attrs["num_samples"])):  
            qpos = demo_data["obs/qpos"][i]  
            qvel = demo_data["obs/qvel"][i]   
            action = demo_data["actions"][i]  
              
            # Load and preprocess images with working size  
            left_img = cv2.imdecode(demo_data["obs/leftImg"][i], cv2.IMREAD_COLOR)  
            right_img = cv2.imdecode(demo_data["obs/rightImg"][i], cv2.IMREAD_COLOR)  
            top_img = cv2.imdecode(demo_data["obs/topImg"][i], cv2.IMREAD_COLOR)  
              
            left_processed = preprocess_image_for_robomimic(left_img, working_size)  
            right_processed = preprocess_image_for_robomimic(right_img, working_size)    
            top_processed = preprocess_image_for_robomimic(top_img, working_size)  
              
            obs_dict = {  
                'qpos': qpos,  
                'qvel': qvel,  
                'leftImg': left_processed,  
                'rightImg': right_processed,  
                'topImg': top_processed  
            }  
              
            predicted_action = model(obs_dict)  
              
            print(f"\nFrame {i}:")  
            print("Ground truth action (deg):", [np.rad2deg(j) for j in action])  
            print("Predicted action (deg):", [np.rad2deg(j) for j in predicted_action])  
            print("Action delta (deg):", [np.rad2deg(j) for j in (predicted_action - action)])  
              
    cv2.destroyAllWindows()