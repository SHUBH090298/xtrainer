import sys  
import os  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ModelTrain/"  
sys.path.append(BASE_DIR)  
  
import cv2  
import time  
from dataclasses import dataclass  
import numpy as np  
import tyro  
import threading  
  
from dobot_control.env import RobotEnv  
from dobot_control.robots.robot_node import ZMQClientRobot  
from dobot_control.cameras.realsense_camera import RealSenseCamera  
from scripts.manipulate_utils import load_ini_data_camera  
  
# Add robomimic imports  
import robomimic.utils.file_utils as FileUtils  
import robomimic.utils.torch_utils as TorchUtils  
  
@dataclass  
class Args:  
    robot_port: int = 6001  
    hostname: str = "127.0.0.1"  
    show_img: bool = True  
    model_type: str = "robomimic"  # or "custom"  
    ckpt_path: str = "/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250917115237/models/model_epoch_50.pth"  
  
# Globals for camera images  
image_left, image_right, image_top, thread_run = None, None, None, None  
lock = threading.Lock()  
  
def run_thread_cam(rs_cam, which_cam):  
    """Camera thread with proper error handling"""  
    global image_left, image_right, image_top, thread_run  
    while thread_run:  
        try:  
            img, _ = rs_cam.read()  
            img = img[:, :, ::-1].copy()  # BGR to RGB and ensure contiguous  
              
            if which_cam == 0:  
                image_left = img  
            elif which_cam == 1:  
                image_right = img  
            elif which_cam == 2:  
                image_top = img  
        except Exception as e:  
            print(f"Camera {which_cam} error: {e}")  
            time.sleep(0.1)  
  
def initialize_robot_safely(env):  
    """Initialize robot to safe positions"""  
    print("Moving to safe position...")  
      
    # Safe position  
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])  
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])  
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])  
      
    curr_joints = env.get_obs()["joint_positions"]  
    max_delta = (np.abs(curr_joints - reset_joints)).max()  
    steps = min(int(max_delta / 0.001), 150)  
      
    for jnt in np.linspace(curr_joints, reset_joints, steps):  
        env.step(jnt, np.array([1, 1]))  
    time.sleep(1)  
      
    # Initial photo position  
    reset_joints_left  = np.deg2rad([-90, 20, -100, 20, 90, 70, 0])
    reset_joints_right = np.deg2rad([90, -20, 100, -20, -90, -70, 0]) 
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])  
      
    curr_joints = env.get_obs()["joint_positions"]  
    max_delta = (np.abs(curr_joints - reset_joints)).max()  
    steps = min(int(max_delta / 0.001), 150)  
      
    for jnt in np.linspace(curr_joints, reset_joints, steps):  
        env.step(jnt, np.array([1, 1]))  
      
    print("Robot initialization complete")  
  
def check_safety_limits(action, last_action, env, t):  
    """Comprehensive safety checking with proper initial movement handling"""  
    protect_err = False  
      
    # Check for large action changes  
    delta = action - last_action  
    max_joint_delta = max(np.abs(delta[0:6]).max(), np.abs(delta[7:13]).max())  
      
    if max_joint_delta > 0.17:  # ~10 degrees  
        print(f"[Warn]: Large joint increment detected: {max_joint_delta:.3f}")  
        print("Joint deltas:", delta)  
          
        # For the very first action (t=0), allow larger movements but execute gradually  
        if t == 0:  
            print("Initial large movement detected - executing gradually...")  
            steps = min(int(max_joint_delta / 0.001), 200)  # More steps for safety  
            for jnt in np.linspace(last_action, action, steps):  
                env.step(jnt, np.array([1, 1]))  
            return False, action  # No error, action executed gradually  
        elif t > 100:  # Allow larger movements after initial period  
            # Gradual movement for large deltas  
            steps = min(int(max_joint_delta / 0.001), 100)  
            for jnt in np.linspace(last_action, action, steps):  
                env.step(jnt, np.array([1, 1]))  
            return False, action  # No error, but action was executed gradually  
        else:  
            # Interactive confirmation for early steps (1-100)  
            print("Do you want to continue? Press 'Y' to continue, any other key to stop.")  
            key = input().strip().lower()  
              
            if key == 'y':  
                steps = min(int(max_joint_delta / 0.001), 150)  
                for jnt in np.linspace(last_action, action, steps):  
                    env.step(jnt, np.array([1, 1]))  
                return False, action  
            else:  
                protect_err = True  
      
    # Joint angle limitations  
    if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and  
            (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):  
        print("[Warn]: J3 or J4 joints out of safe position!")  
        print("Action:", action)  
        protect_err = True  
      
    # Cartesian position limits  
    try:  
        pos = env.get_XYZrxryrz_state()  
        if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 42) and  
                (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 42)):  
            print("[Warn]: Robot arm XYZ out of safe position!")  
            print("Position:", pos)  
            protect_err = True  
    except Exception as e:  
        print(f"Error checking position limits: {e}")  
        protect_err = True  
      
    return protect_err, action  
  
def preprocess_image_for_robomimic(image):  
    """Preprocess image for robomimic models"""  
    if image.shape != (480, 640, 3):  
        image = cv2.resize(image, (640, 480))  
    image = image.astype(np.float32) / 255.0  
    image = np.transpose(image, (2, 0, 1))  
    return np.ascontiguousarray(image)  
  
def main(args: Args):  
    global image_left, image_right, image_top, thread_run  
    thread_run = True  
  
    # Initialize cameras  
    camera_dict = load_ini_data_camera()  
    rs_cam_left = RealSenseCamera(flip=True, device_id=camera_dict["left"])  
    rs_cam_right = RealSenseCamera(flip=False, device_id=camera_dict["right"])  
    rs_cam_top = RealSenseCamera(flip=False, device_id=camera_dict["top"])  
  
    # Start camera threads  
    threads = [  
        threading.Thread(target=run_thread_cam, args=(rs_cam_left, 0), daemon=True),  
        threading.Thread(target=run_thread_cam, args=(rs_cam_right, 1), daemon=True),  
        threading.Thread(target=run_thread_cam, args=(rs_cam_top, 2), daemon=True)  
    ]  
      
    for thread in threads:  
        thread.start()  
      
    print("Camera threads initialized...")  
  
    # Initialize robot  
    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)  
    env = RobotEnv(robot)  
      
    # Set status lights  
    env.set_do_status([1, 0])  # Red off  
    env.set_do_status([2, 0])  # Green off    
    env.set_do_status([3, 0])  # Yellow off  
      
    print("Robot initialized...")  
  
    # Initialize robot to safe position  
    initialize_robot_safely(env)  
  
    # Load model based on type  
    if args.model_type == "robomimic":  
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)  
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(  
            ckpt_path=args.ckpt_path, device=device, verbose=True  
        )  
        print("Robomimic model loaded...")  
    else:  
        from ModelTrain.module.model_module import Imitate_Model  
        model = Imitate_Model(ckpt_dir='./ckpt/ckpt_move_cube_new', ckpt_name='policy_last.ckpt')  
        model.loadModel()  
        print("Custom model loaded...")  
  
    # Wait for cameras to stabilize  
    time.sleep(2)  
  
    # Main control loop  
    obs = env.get_obs()  
    obs["joint_positions"][6] = 1.0   # Initialize gripper positions  
    obs["joint_positions"][13] = 1.0  
    last_action = obs["joint_positions"].copy()  
    first = True  
    t = 0  
    episode_len = 900  
  
    print("Starting autonomous operation...")  
    env.set_do_status([3, 1])  # Yellow light on (active)  
  
    try:  
        while t < episode_len:  
            time0 = time.time()  
  
            # Read images safely  
            with lock:  
                current_image_left = image_left.copy() if image_left is not None else np.zeros((480, 640, 3), dtype=np.uint8)  
                current_image_right = image_right.copy() if image_right is not None else np.zeros((480, 640, 3), dtype=np.uint8)  
                current_image_top = image_top.copy() if image_top is not None else np.zeros((480, 640, 3), dtype=np.uint8)  
  
            # Get action from model  
            if args.model_type == "robomimic":  
                observation = {  
                    'qpos': np.ascontiguousarray(obs["joint_positions"].astype(np.float64)),  
                    'qvel': np.ascontiguousarray(np.zeros_like(obs["joint_positions"]).astype(np.float64)),  
                    'leftImg': preprocess_image_for_robomimic(current_image_left),  
                    'rightImg': preprocess_image_for_robomimic(current_image_right),  
                    'topImg': preprocess_image_for_robomimic(current_image_top)  
                }  
                  
                action_tensor = policy(observation)  
                if hasattr(action_tensor, 'detach'):  
                    action = action_tensor.detach().cpu().numpy()  
                else:  
                    action = np.array(action_tensor)  
                  
                if action.shape == (1, 14):  
                    action = action.squeeze(0)  
            else:  
                observation = {  
                    'qpos': obs["joint_positions"],  
                    'images': {  
                        'left_wrist': current_image_left,  
                        'right_wrist': current_image_right,  
                        'top': current_image_top  
                    }  
                }  
                action = model.predict(observation, t)  
  
            # Clamp gripper values  
            action[6] = np.clip(action[6], 0, 1)  
            action[13] = np.clip(action[13], 0, 1)  
  
            # Safety checks  
            protect_err, processed_action = check_safety_limits(action, last_action, env, t)  
              
            if protect_err:  
                env.set_do_status([3, 0])  # Yellow off  
                env.set_do_status([2, 0])  # Green off  
                env.set_do_status([1, 1])  # Red on  
                print("Safety violation - stopping")  
                break  
  
            # Execute action (skip the first action execution since it's handled in safety check)  
            if not first and t > 0:  
                obs = env.step(processed_action, np.array([1, 1]))  
                # Update gripper positions in observation  
                obs["joint_positions"][6] = processed_action[6]  
                obs["joint_positions"][13] = processed_action[13]  
            elif first:  
                # For first action, just update observation without additional movement  
                obs = env.get_obs()  
                obs["joint_positions"][6] = processed_action[6]  
                obs["joint_positions"][13] = processed_action[13]  
                first = False  
  
            # Display images  
            if args.show_img:  
                show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)  
                show_canvas[:, :640] = current_image_left  
                show_canvas[:, 640:1280] = current_image_right  
                show_canvas[:, 1280:] = current_image_top  
                cv2.imshow("Camera Views", show_canvas)  
                  
                if cv2.waitKey(1) & 0xFF == ord('q'):  
                    break  
  
            last_action = processed_action.copy()  
            t += 1  
              
            print(f"Step {t}, Total time: {(time.time() - time0) * 1000:.1f}ms")  
  
    except Exception as e:  
        print(f"Error in main loop: {e}")  
        import traceback  
        traceback.print_exc()  
      
    finally:  
        thread_run = False  
        env.set_do_status([2, 1])  # Green light on (completed)  
        cv2.destroyAllWindows()  
        print("Task completed")  
  
if __name__ == "__main__":  
    main(tyro.cli(Args))