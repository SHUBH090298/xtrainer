from ModelTrain.module.model_module import Imitate_Model
import cv2
import torch

import h5py
import numpy as np
import cv2


if __name__ == '__main__':
    # model = Imitate_Model(ckpt_dir='./ckpt/ckpt_move_cube_new',ckpt_name='policy_last.ckpt')
    model = Imitate_Model(ckpt_name='policy_last.ckpt')
    model.loadModel()
    # Load your checkpoint and check for action normalization stats  
    ckpt_dict = torch.load("./ckpt/ckpt_move_cube_new/policy_last.ckpt")  
    if "action_normalization_stats" in ckpt_dict:  
        print("Action normalization stats found:", ckpt_dict["action_normalization_stats"])  
        # Extract scale and offset for denormalization
        action_mean = np.array(ckpt_dict["action_normalization_stats"]["mean"])
        action_std = np.array(ckpt_dict["action_normalization_stats"]["std"])
    else:  
        print("ERROR: No action normalization stats in checkpoint!")
        action_mean = None
        action_std = None

    observation = {'qpos':[],'images':{'left_wrist':[],'right_wrist':[],'top':[]}}
    i=0
    # while i<10:
        # observation['qpos'] = [-1.57, 0, -1.57, 0, 1.57, 1.57, 1, 1.57, 0, 1.57, 0, -1.57, -1.57, 1]  #  input joint value (unit radians) and Grippers value(0~1).The 7th and 14th values are the left and right hand gripper values, respectively
        # observation['images']['left_wrist'] = cv2.imread("./testimg/left_wrist.jpg", 1)  # input image
        # observation['images']['right_wrist'] = cv2.imread("./testimg/right_wrist.jpg", 1)
        # observation['images']['top'] = cv2.imread("./testimg/top.jpg", 1)
        # cv2.imshow("img", observation['images']['left_wrist'])
        # cv2.waitKey(0)
        # action = model.predict(observation,i)  # out put
        # print(action)
        # i +=1

    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    with h5py.File("/home/shubh/xtrainer/experiments/datasets/Pick_Place/train_data/episode_init_16.hdf5", 'r', rdcc_nbytes=1024 ** 2 * 2) as root:
        print(len(root["/observations/images/top"]))
        for i in range(len(root["/observations/images/top"])):
            qpos = root["/observations/qpos"][i]
            print("qpos:",[np.rad2deg(i) for i in qpos])
            action = root["action"][i]
            print("action:",[np.rad2deg(i) for i in action])
            show_canvas[:, :640] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR), dtype="uint8")
            cv2.imshow("0", show_canvas)

            observation['qpos'] = qpos  #  input joint value (unit radians) and Grippers value(0~1).The 7th and 14th values are the left and right hand gripper values, respectively
            observation['images']['left_wrist'] = cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)  # input image
            observation['images']['right_wrist'] = cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)
            observation['images']['top'] = cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR)

            predict_action = model.predict(observation, i)  # output
            print("Raw model output range:", predict_action.min(), "to", predict_action.max())  
            print("Raw model output shape:", predict_action.shape)
            
            # Extract the first timestep and first 14 dimensions from the predicted action sequence  
            current_action = predict_action[0, :14]  # Take first timestep, first 14 dimensions  

            # Denormalize predicted action if stats are available
            if action_mean is not None and action_std is not None:
                current_action = current_action * action_std + action_mean

            print("action_delta:",[np.rad2deg(i) for i in (current_action-action)])
            i += 1
            cv2.waitKey(0)
