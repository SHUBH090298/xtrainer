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
import queue

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

# Globals for camera images
image_left, image_right, image_top, thread_run = None, None, None, None
lock = threading.Lock()

def run_thread_cam(rs_cam, which_cam):
    """
    Camera thread reading loop.
    which_cam: 0->left, 1->right, 2->top
    """
    global image_left, image_right, image_top, thread_run
    if which_cam == 0:
        while thread_run:
            img, _ = rs_cam.read()
            # Fix: Use copy() to avoid negative strides and convert BGR->RGB via [:,:,::-1]
            try:
                img = img[:, :, ::-1].copy()
            except Exception:
                img = np.array(img).copy()
            image_left = img
    elif which_cam == 1:
        while thread_run:
            img, _ = rs_cam.read()
            try:
                img = img[:, :, ::-1].copy()
            except Exception:
                img = np.array(img).copy()
            image_right = img
    elif which_cam == 2:
        while thread_run:
            img, _ = rs_cam.read()
            try:
                img = img[:, :, ::-1].copy()
            except Exception:
                img = np.array(img).copy()
            image_top = img

def preprocess_image_for_model(image):
    """
    Keep for legacy; ensures image is (480,640,3) uint8 contiguous.
    """
    # Ensure the image is the right size (480, 640, 3)
    if image.shape != (480, 640, 3):
        image = cv2.resize(image, (640, 480))
    # Ensure uint8 format and contiguous memory
    image = np.ascontiguousarray(image.astype(np.uint8))
    return image

def preprocess_image_for_robomimic(image):
    """
    Preprocess image to match robomimic's expected format:
    - Resize to (480,640,3) if needed
    - Normalize to [0,1] float32
    - Transpose to (C, H, W)
    - Return contiguous array
    """
    # Ensure image is (480, 640, 3) uint8
    if image.shape != (480, 640, 3):
        image = cv2.resize(image, (640, 480))
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Convert from (H, W, C) to (C, H, W)
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
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs_cam_left, 0), daemon=True)
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs_cam_right, 1), daemon=True)
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs_cam_top, 2), daemon=True)

    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_top.start()

    print("camera thread init success...")

    # Initialize robot
    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot)
    print("robot init success...")

    # Load robomimic model
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_path = "/home/shubh/xtrainer/robomimic-r2d2/bc_trained_models/test/20250917115237/models/model_epoch_50.pth"
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    print("model init success...")
    print("The robot begins to perform tasks autonomously...")

    # Wait for cameras to initialize
    time.sleep(2)

    obs = env.get_obs()
    last_action = np.zeros(14)
    first = True
    t = 0

    try:
        while True:
            time0 = time.time()

            # Read images
            time1 = time.time()
            with lock:
                current_image_left = image_left.copy() if image_left is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                current_image_right = image_right.copy() if image_right is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                current_image_top = image_top.copy() if image_top is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            time2 = time.time()
            print("read images time(ms)：", (time2 - time1) * 1000)

            # Build observation for model
            observation = {
                'qpos': np.ascontiguousarray(obs["joint_positions"].astype(np.float64)),
                'qvel': np.ascontiguousarray(np.zeros_like(obs["joint_positions"]).astype(np.float64)),
                'leftImg': preprocess_image_for_robomimic(current_image_left),
                'rightImg': preprocess_image_for_robomimic(current_image_right),
                'topImg': preprocess_image_for_robomimic(current_image_top)
            }

            # Add debugging code right before calling the policy
            print("=== DEBUGGING IMAGE SHAPES ===")
            shape_meta = ckpt_dict.get("shape_metadata", {})
            print("Expected shapes from model:")
            all_shapes = shape_meta.get("all_shapes", {})
            for key in ['leftImg', 'rightImg', 'topImg']:
                if key in all_shapes:
                    print(f"  {key}: {all_shapes[key]}")

            print("Actual observation shapes:")
            for key, obs_val in observation.items():
                if hasattr(obs_val, 'shape'):
                    print(f"  {key}: {obs_val.shape}, dtype: {obs_val.dtype}")
                else:
                    print(f"  {key}: {type(obs_val)}")

            # Also check what the visual encoder expects
            try:
                visual_cores = []
                # Try to inspect policy nets for modules that have input_shape attribute
                if hasattr(policy, "nets"):
                    for name, module in policy.nets["policy"].named_modules():
                        if hasattr(module, 'input_shape'):
                            visual_cores.append((name, module.input_shape))
                print("Visual encoder input shapes:")
                for name, shape in visual_cores:
                    print(f"  {name}: {shape}")
            except Exception as e:
                print(f"Could not inspect visual encoders: {e}")

            try:
                # Get action from policy
                action_tensor = policy(observation)

                # Convert to numpy and handle dimensions
                if hasattr(action_tensor, 'detach'):
                    action = action_tensor.detach().cpu().numpy()
                else:
                    action = np.array(action_tensor)

                # Handle action dimensions - your model outputs 14D actions
                if action.shape == (14,):
                    pass  # Already correct shape
                elif action.shape == (1, 14):
                    action = action.squeeze(0)
                else:
                    print(f"Unexpected action shape: {action.shape}")
                    action = np.zeros(14)

            except Exception as e:
                print(f"Error during model inference: {e}")
                import traceback
                traceback.print_exc()
                break

            # Display images if requested
            if args.show_img:
                try:
                    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
                    show_canvas[:, :640] = current_image_left
                    show_canvas[:, 640:1280] = current_image_right
                    show_canvas[:, 1280:] = current_image_top
                    cv2.imshow("Camera Views", show_canvas)
                    # small wait to allow image to be shown; returns key pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested via GUI keypress.")
                        break
                except Exception as e:
                    print("Error displaying images:", e)

            # Safety checks
            protect_err = False

            # Check for large action changes
            if not first:
                if np.max(np.abs(action - last_action)) > 0.3:
                    print("[Warn]: The action change is too large!")
                    print("Current action:", action)
                    print("Last action:", last_action)
                    print("Action delta:", action - last_action)
                    if t > 100:
                        obs = env.step(action, np.array([1, 1]))
                        first = False
                    else:
                        protect_err = True
                        cv2.destroyAllWindows()

            # Joint angle limitations
            try:
                if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and
                        (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
                    print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
                    print(action)
                    protect_err = True
            except Exception as e:
                print("Error checking joint limits:", e)
                protect_err = True

            if protect_err:
                break

            if not first:
                time3 = time.time()
                obs = env.step(action, np.array([1, 1]))
                time4 = time.time()

                # Update observations (preserve original structure)
                try:
                    obs["joint_positions"][6] = action[6] if len(action) > 6 else obs["joint_positions"][6]
                    obs["joint_positions"][13] = action[13] if len(action) > 13 else obs["joint_positions"][13]
                except Exception as e:
                    print("Error updating obs joint positions from action:", e)

                print("Read joint value time(ms)：", (time4 - time3) * 1000)
                t += 1
                print("The total time(ms):", (time4 - time0) * 1000)

            last_action = action.copy()
            first = False

    finally:
        thread_run = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Task finished / thread stopped.")

if __name__ == "__main__":
    main(tyro.cli(Args))
