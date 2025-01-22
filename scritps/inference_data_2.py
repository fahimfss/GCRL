import gymnasium as gym 
import numpy as np
import os 
import time
import cv2
import warnings 
warnings.filterwarnings("ignore")

from jsac.envs.rl_chemist.env import RLC_Env
from jsac.helpers.utils import WrappedEnv

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='FrankaEnv-v0', type=str)
    parser.add_argument('--env_mode', default='inference_1', type=str) 
    parser.add_argument('--target_obj_num', default=7, type=int) 
    parser.add_argument('--image_height', default=480, type=int)     # Mode: img, img_prop
    parser.add_argument('--image_width', default=640, type=int)      # Mode: img, img_prop     
    parser.add_argument('--image_history', default=1, type=int)      # Mode: img, img_prop

    args = parser.parse_args()
    return args


def main(env_mode, target_obj_num, itr):
    args = parse_args()
    args.target_obj_num = target_obj_num
    args.env_mode = env_mode

    env = RLC_Env(args.env_name, 
                  args.image_history, 
                  args.image_width, 
                  args.image_height, 
                  args.env_mode, 
                  "ground_truth",
                  args.target_obj_num)
    env = WrappedEnv(env, 200)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space 
    
    print('Action shape:', args.action_shape)

    trial = 1

    folder_name='inference_test_data_1'
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
        
    images_folder=f'images_{itr}'
    if not os.path.exists(f'{folder_name}/{images_folder}'): 
        os.makedirs(f'{folder_name}/{images_folder}')

    fl = open(f'{folder_name}/info_{itr}.txt', "a")

    img_itr = len(os.listdir(f'{folder_name}/{images_folder}'))

    start_0 = 0.4
    change = 0.3
    flag = False

    for i in range(trial): 
        ep_rewards = 0
        solved, done = False, False
        state = env.reset(create_vid=False) 
            
        step = 0
        if i >= 3:
            flag = True

        while not solved and step < 150:
            if step < 5: 
                state, reward, done, info = env.step(np.zeros(args.action_shape, dtype=np.float32))
            else:
                action = np.array([0, start_0, 0, 0.23, 0.0, 0.12, 0.0, 0.0])
                        
                t1 = time.time()
                state, reward, done, info = env.step(action)
                t2 = time.time()
                
                img = state[0].copy()  
                img = (img[..., :3] * 0.7 + img[..., 3:]*0.3).astype(img.dtype)
                 
                cv2.imshow("img", img)
                cv2.waitKey(80) 
        
            solved = done
            step += 1

    fl.close()


if __name__ == '__main__':
    for i in range(20):
        main('inference_1', i, 1)