from ModelTrain.module.model_module import Imitate_Model
import cv2
import h5py
import numpy as np

if __name__ == '__main__':
    # Initialize and load the model
    model = Imitate_Model(ckpt_dir='./ckpt/ckpt_move_cube_new', ckpt_name='policy_last.ckpt')
    model.loadModel()

    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)

    dataset_path = "/home/shubh/dobot_xtrainer/experiments/datasets/Pick_Place/train_data/episode_init_46.hdf5"

    with h5py.File(dataset_path, 'r', rdcc_nbytes=1024 ** 2 * 2) as root:
        total_frames = len(root["/observations/images/top"])
        print(f"Total frames: {total_frames}")

        for i in range(total_frames):
            # Read joint positions
            qpos = root["/observations/qpos"][i]
            print("qpos (deg):", [np.rad2deg(j) for j in qpos])

            # Read ground-truth action
            action = root["action"][i]
            print("action (deg):", [np.rad2deg(j) for j in action])

            # Decode and display images
            top_img = cv2.imdecode(np.asarray(root["/observations/images/top"][i], dtype="uint8"), cv2.IMREAD_COLOR)
            left_img = cv2.imdecode(np.asarray(root["/observations/images/left_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)
            right_img = cv2.imdecode(np.asarray(root["/observations/images/right_wrist"][i], dtype="uint8"), cv2.IMREAD_COLOR)

            show_canvas[:, :640] = top_img
            show_canvas[:, 640:640 * 2] = left_img
            show_canvas[:, 640 * 2:640 * 3] = right_img
            cv2.imshow("Observation", show_canvas)

            # Prepare observation dictionary for model
            observation['qpos'] = qpos
            observation['images']['top'] = top_img
            observation['images']['left_wrist'] = left_img
            observation['images']['right_wrist'] = right_img

            # Predict action
            predict_action_full = model.predict(observation, i)  # may have shape (N, M)
            

            # Ensure predict_action matches action shape
            predict_action = np.array(predict_action_full).flatten()[:len(action)]
            print("predicted ",predict_action)
            # Compute difference in degrees
            action_delta = np.rad2deg(predict_action - action)
            print("action_delta (deg):", action_delta)

            cv2.waitKey(0)

    cv2.destroyAllWindows()
