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
  
# Add robomimic imports  
import robomimic.utils.file_utils as FileUtils  
import robomimic.utils.torch_utils as TorchUtils  
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
            image_left = image_left[:, :, ::-1]  
    elif which_cam==1:  
        while thread_run:  
            image_right, _ = rs_cam.read()  
            image_right = image_right[:, :, ::-1]  
    elif which_cam==2:  
        while thread_run:  
            image_top, _ = rs_cam.read()  
            image_top = image_top[:, :, ::-1]  
    else:  
        print("Camera index error! ")  
  
def preprocess_image_for_model(image):  
    """Preprocess camera image to match training format"""  
    # Ensure image is the right size (480, 640, 3) and uint8  
    if image.shape != (480, 640, 3):  
        image = cv2.resize(image, (640, 480))  
      
    # Ensure uint8 format and contiguous memory  
    image = np.ascontiguousarray(image.astype(np.uint8))  
      
    # Keep in (H, W, C) format - let robomimic handle the conversion  
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
  
    # Initialize the robomimic model  
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)  
    ckpt_path = "/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250916113358/models/model_epoch_50.pth"  
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
          
        # Preprocess images to match training format  
        processed_left = preprocess_image_for_model(image_left)  
        processed_right = preprocess_image_for_model(image_right)  
        processed_top = preprocess_image_for_model(image_top)  
          
        # Create observation dictionary matching training format  
        observation = {  
            'qpos': np.ascontiguousarray(obs["joint_positions"].astype(np.float64)),  
            'qvel': np.ascontiguousarray(np.zeros_like(obs["joint_positions"]).astype(np.float64)),  
            'rightImg': processed_right,  
            'topImg': processed_top,  
            'leftImg': processed_left  
        }  
          
        if args.show_img:  
            imgs = np.hstack((processed_left, processed_right, processed_top))  
            cv2.imshow("imgs", imgs)  
            cv2.waitKey(1)  
        time1 = time.time()  
        print("read images time(ms)：",(time1-time0)*1000)  
  
        try:  
            # Model inference  
            action_tensor = policy(observation)  
              
            # Convert tensor to numpy array  
            if torch.is_tensor(action_tensor):  
                action = action_tensor.detach().cpu().numpy()  
            else:  
                action = np.array(action_tensor)  
              
            # Handle action shape - expand to 14 dimensions if needed  
            if action.shape == (1,):  
                # Single action output - need to map to full joint space  
                action_full = obs["joint_positions"].copy()  
                # Map the single action to appropriate joint(s) based on your task  
                # This is task-specific - you may need to adjust this mapping  
                action_full[0] = action[0]  # Example: map to first joint  
                action = action_full  
            elif len(action.shape) == 1 and action.shape[0] < 14:  
                # Partial action vector - pad with current joint positions  
                action_full = obs["joint_positions"].copy()  
                action_full[:len(action)] = action  
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