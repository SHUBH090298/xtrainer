import sys  
import os  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/ModelTrain/"  
sys.path.append(BASE_DIR)  
import cv2  
import time  
from dataclasses import dataclass  
import numpy as np  
import tyro  
import threading  
import queue  
from dobot_control.env import RobotEnv  
from dobot_control.robots.robot_node import ZMQClientRobot  
from dobot_control.cameras.realsense_camera import RealSenseCamera  
  
from scripts.manipulate_utils import load_ini_data_camera  
  
# Use robomimic imports for robomimic checkpoints  
import robomimic.utils.file_utils as FileUtils  
import robomimic.utils.torch_utils as TorchUtils  
import robomimic.utils.obs_utils as ObsUtils  
import torch  
  
@dataclass  
class Args:  
    robot_port: int = 6001  
    hostname: str = "127.0.0.1"  
    show_img: bool = True  
  
image_left,image_right,image_top,thread_run=None,None,None,None  
lock = threading.Lock()  
  
def run_thread_cam(rs_cam, which_cam):  
    global image_left, image_right, image_top, thread_run  
    if which_cam==0:  
        while thread_run:  
            image_left, _ = rs_cam.read()  
            # Fix: Use copy() to avoid negative strides  
            image_left = image_left[:, :, ::-1].copy()  
    elif which_cam==1:  
        while thread_run:  
            image_right, _ = rs_cam.read()  
            # Fix: Use copy() to avoid negative strides  
            image_right = image_right[:, :, ::-1].copy()  
    elif which_cam==2:  
        while thread_run:  
            image_top, _ = rs_cam.read()  
            # Fix: Use copy() to avoid negative strides  
            image_top = image_top[:, :, ::-1].copy()  
    else:  
        print("Camera index error! ")  
  
def preprocess_image_for_robomimic(image):  
    """Preprocess camera image for robomimic model"""  
    # Ensure correct size (480, 640, 3)  
    if image.shape != (480, 640, 3):  
        image = cv2.resize(image, (640, 480))  
      
    # Convert to float32 and normalize to [0, 1]  
    image = image.astype(np.float32) / 255.0  
      
    # Convert from (H, W, C) to (C, H, W) format  
    image = np.transpose(image, (2, 0, 1))  
      
    return image  
  
def main(args):  
    # camera init  
    global image_left, image_right, image_top, thread_run  
    thread_run=True  
    camera_dict = load_ini_data_camera()  
    rs1 = RealSenseCamera(flip=False, device_id=camera_dict["left"])  
    rs2 = RealSenseCamera(flip=True, device_id=camera_dict["right"])  
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["top"])  
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs1, 0))  
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs2, 1))  
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs3, 2))  
    thread_cam_left.start()  
    thread_cam_right.start()  
    thread_cam_top.start()  
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)  
    time.sleep(2)  
    print("camera thread init success...")  
  
    # robot init  
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)  
    env = RobotEnv(robot_client)  
    env.set_do_status([1, 0])  
    env.set_do_status([2, 0])  
    env.set_do_status([3, 0])  
    print("robot init success...")  
  
    # go to the safe position  
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])  
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])  
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])  
    curr_joints = env.get_obs()["joint_positions"]  
    max_delta = (np.abs(curr_joints - reset_joints)).max()  
    steps = min(int(max_delta / 0.001), 150)  
    for jnt in np.linspace(curr_joints, reset_joints, steps):  
        env.step(jnt,np.array([1,1]))  
    time.sleep(1)  
  
    # go to the initial photo position  
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])  
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])  
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])  
    curr_joints = env.get_obs()["joint_positions"]  
    max_delta = (np.abs(curr_joints - reset_joints)).max()  
    steps = min(int(max_delta / 0.001), 150)  
    for jnt in np.linspace(curr_joints, reset_joints, steps):  
        env.step(jnt,np.array([1,1]))  
  
    # Initialize robomimic model properly  
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)  
    ckpt_path = "/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250916113358/models/model_epoch_50.pth"  
      
    # Load policy using robomimic's method  
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)  
    print("model init success...")  
  
    # Initialize the parameters  
    episode_len = 900  
    t=0  
    last_time = 0  
    obs = env.get_obs()  
    obs["joint_positions"][6] = 1.0  
    obs["joint_positions"][13] = 1.0  
    last_action = obs["joint_positions"].copy()  
  
    first = True  
  
    print("The robot begins to perform tasks autonomously...")  
    while t < episode_len:  
        # Obtain the current images  
        time0 = time.time()  
          
        # Preprocess images to match model expectations  
        processed_left = preprocess_image_for_robomimic(image_left)  
        processed_right = preprocess_image_for_robomimic(image_right)  
        processed_top = preprocess_image_for_robomimic(image_top)  
          
        # Create observation dictionary using the exact keys from config  
        # Note: qpos and qvel shapes are [1] according to the model config  
        observation = {  
            'qpos': np.array([obs["joint_positions"][0]], dtype=np.float32),  # Single value  
            'qvel': np.array([0.0], dtype=np.float32),  # Single velocity value  
            'leftImg': processed_left,  
            'rightImg': processed_right,  
            'topImg': processed_top  
        }  
          
        if args.show_img:  
            imgs = np.hstack((image_left, image_right, image_top))  
            cv2.imshow("imgs", imgs)  
            cv2.waitKey(1)  
        time1 = time.time()  
        print("read images time(ms)：",(time1-time0)*1000)  
  
        try:  
            # Model inference using robomimic policy  
            action = policy(observation)  
              
            # Convert to numpy if needed  
            if torch.is_tensor(action):  
                action = action.detach().cpu().numpy()  
              
            # Handle single action output (action_dim=1 from config)  
            if action.shape == (1,) or len(action) == 1:  
                # This model only predicts 1 action dimension  
                # Map this to your robot's control scheme  
                action_full = obs["joint_positions"].copy()  
                  
                # Example mapping - adjust based on your task  
                # You might want to map to gripper, specific joint, or scale to multiple joints  
                action_full[6] = np.clip(action[0], 0, 1)  # Map to left gripper  
                action_full[13] = np.clip(action[0], 0, 1)  # Map to right gripper  
                action = action_full  
              
            # Gripper constraints  
            if len(action) > 6:  
                action[6] = np.clip(action[6], 0, 1)  
            if len(action) > 13:  
                action[13] = np.clip(action[13], 0, 1)  
                  
        except Exception as e:  
            print(f"Error during model inference: {e}")  
            import traceback  
            traceback.print_exc()  
            break  
              
        time2 = time.time()  
        print("Model inference time(ms)：", (time2 - time1) * 1000)  
  
        # Safety protection  
        protect_err = False  
  
        # Velocity protection  
        if not first:  
            max_delta = (np.abs(last_action - action)).max()  
            if max_delta > 0.1:  
                print("[Warn]:The action change is too large! max delta:", max_delta)  
                temp_img = np.zeros(shape=(640, 480))  
                cv2.imshow("waitKey", temp_img)  
                key = cv2.waitKey(0)  
                if key == ord('y') or key == ord('Y'):  
                    cv2.destroyWindow("waitKey")  
                    max_delta = (np.abs(last_action - action)).max()  
                    steps = min(int(max_delta / 0.001), 100)  
                    for jnt in np.linspace(last_action, action, steps):  
                        env.step(jnt,np.array([1,1]))  
                    first = False  
                else:  
                    protect_err = True  
                    cv2.destroyAllWindows()  
  
        # Joint angle limitations  
        if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and   
                (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):  
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")  
            print(action)  
            protect_err = True  
  
        if protect_err:  
            break  
  
        if not first:  
            time3 = time.time()  
            obs = env.step(action,np.array([1,1]))  
            time4 = time.time()  
  
            # Update observations  
            obs["joint_positions"][6] = action[6] if len(action) > 6 else obs["joint_positions"][6]  
            obs["joint_positions"][13] = action[13] if len(action) > 13 else obs["joint_positions"][13]  
  
            print("Read joint value time(ms)：", (time4 - time3) * 1000)  
            t +=1  
            print("The total time(ms):", (time4 - time0) * 1000)  
  
        last_action = action.copy()  
        first = False  
  
    thread_run = False  
    print("Task accomplished")  
  
if __name__ == "__main__":  
    main(tyro.cli(Args))