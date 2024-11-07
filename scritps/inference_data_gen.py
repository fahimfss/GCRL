import gymnasium as gym 
import numpy as np
import os 
import time
import cv2
import warnings 
warnings.filterwarnings("ignore")

from jsac.envs.rl_chemist.ur10 import UR10_ENV
from jsac.helpers.utils import WrappedEnv

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='UR10eEnv-v0', type=str)
    parser.add_argument('--env_mode', default='inference_1', type=str) 
    parser.add_argument('--target_obj_num', default=7, type=int) 
    parser.add_argument('--image_height', default=480, type=int)     # Mode: img, img_prop
    parser.add_argument('--image_width', default=848, type=int)      # Mode: img, img_prop     
    parser.add_argument('--image_history', default=1, type=int)      # Mode: img, img_prop

    args = parser.parse_args()
    return args


def main(env_mode, target_obj_num, itr):
    args = parse_args()
    args.target_obj_num = target_obj_num
    args.env_mode = env_mode

    env = UR10_ENV(args.env_name, args.image_history, args.image_width, args.image_height, args.env_mode, args.target_obj_num)
    env = WrappedEnv(env, 200)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space 

    trial = 5

    folder_name='inference_test_data_1'
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
        
    images_folder=f'images_{itr}'
    if not os.path.exists(f'{folder_name}/{images_folder}'): 
        os.makedirs(f'{folder_name}/{images_folder}')

    fl = open(f'{folder_name}/info_{itr}.txt', "a")

    img_itr = len(os.listdir(f'{folder_name}/{images_folder}'))

    start_0 = 0.3
    change = 0.3
    flag = False

    for i in range(trial): 
        ep_rewards = 0
        solved, done = False, False
        if i < 10: 
            state = env.reset(create_vid=False) 
        else:
            state = env.reset(create_vid=False) 
            
        step = 0
        if i >= 3:
            flag = True

        while not solved and step < 200:
            if step < 5: 
                state, reward, done, info = env.step(np.zeros((7,), dtype=np.float32))
            else:
                action = np.array([start_0, -0.65, 0.68, 0.0, 0.0, 0.0, 0.0])
                noise = np.random.uniform(-0.25, 0.25, (7,))
                
                if i == 3:
                    if flag:
                        action[0] = 0.3 
                        state, reward, done, info = env.step(action)
                        step += 1
                        x = info['x']
                        y = info['y']
                
                        if x >= 60 and x < 790 and y >= 50 and y < 430:
                            pass
                        else:
                            flag = False
                    
                        continue
                    else:
                        action = np.array([-0.4, 0.05, 0.3, -0.1, 0.0, 0.0, 0.0])
                        
                if i == 4:
                    if flag:
                        action[0] = -0.3 
                        state, reward, done, info = env.step(action)
                        step += 1
                        x = info['x']
                        y = info['y']
                
                        if x >= 60 and x < 790 and y >= 50 and y < 430:
                            pass
                        else:
                            flag = False
                            
                        continue
                    else:
                        action = np.array([0.4, 0.1, 0.25, -0.1, 0.0, 0.0, 0.0])
                        
                t1 = time.time()
                state, reward, done, info = env.step(action)
                t2 = time.time()
                
                img = state[0][:, :, 0:3].copy() 
                x = info['x']
                y = info['y']
                
                if x >= 25 and x < 825 and y >=0 and y < 480: 
                    # if x < 0:
                    #     x = 0
                    # if x > 799:
                    #     x = 799
                    # if y < 0:
                    #     y = 0
                    # if y > 799:
                    #     y = 799
                    
                    itr_str = str(img_itr).zfill(6)
                    img_itr += 1
                    img_flname = f'{folder_name}/{images_folder}/{itr_str}.png'
                    
                    prompt = info['prompt']
                    distance = info['reach_err']
                    
                    info_str = '{' + f'"img_path": "{img_flname}", "x": {x}, "y": {y}, "distance":{distance}, "prompt": "{prompt}"' + '}\n'
                    fl.write(info_str)
                    fl.flush()
                    
                    # cv2.circle(img, (x, y), 9, (255, 0, 0), -1)
                    # cv2.imshow("img", img)
                    # cv2.waitKey(10)

                    cv2.imwrite(img_flname, img)
        
            step += 1
            
            if i >= 3 and step == 140:
                break
            elif i < 3 and (step == 67 or done or 'truncated' in info):
                start_0 -= change
                break
    

    fl.close()


if __name__ == '__main__':
    for i in range(8):
        print(i)
        main('inference_1', i, 1)
        main('inference_3', i, 3)