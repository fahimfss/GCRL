import gymnasium as gym 
import numpy as np
import os 
import time
import cv2
import warnings 
warnings.filterwarnings("ignore")
from collections import deque

from jsac.envs.rl_chemist.ur10 import UR10_ENV
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from jsac.helpers.utils import WrappedEnv

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")


config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 5, 2],
        [32, 32, 5, 2],
        [32, 64, 3, 2],
        [64, 64, 3, 2], 
    ],
    
    'latent_dim': 128,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='UR10eEnv-v0', type=str)
    parser.add_argument('--env_mode', default='inference_3', type=str) 
    parser.add_argument('--task_name', default='baseline', type=str)
    parser.add_argument('--target_obj_num', default=7, type=int)     # 0 to 7
    parser.add_argument('--image_height', default=480, type=int)     # Mode: img, img_prop
    parser.add_argument('--image_width', default=848, type=int)      # Mode: img, img_prop     
    parser.add_argument('--image_history', default=1, type=int)      # Mode: img, img_prop
    parser.add_argument('--mask_delay_type', default='none', type=str)
    parser.add_argument('--mask_delay_steps', default=2, type=int) 

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=10_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=250_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--global_norm', default=1.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float) 
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=2e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)   # Sync mode: False
    
    # encoder
    parser.add_argument('--spatial_softmax', default=True, action='store_true')    # Mode: img, img_prop

    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=10_000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=25_000, type=int)
    parser.add_argument('--load_model', default=20000, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args


def fn(obj, ckpt, folder_name, images_folder, info_itr):
    args = parse_args()
    args.target_obj_num = obj
    args.load_model = ckpt

    sync_mode = 'sync' if args.sync_mode else 'async'
    args.name = f'{args.env_name}_{args.mode}_{sync_mode}_{args.task_name}'
    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'
    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 

    env = UR10_ENV(args.env_name, args.image_history, args.image_width, args.image_height, args.env_mode, args.target_obj_num)
    env = WrappedEnv(env, 200)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space 

    args.net_params = config

    agent = SACRADAgent(vars(args)) 

    trial = 1

    fl = open(f'{folder_name}/info_{info_itr}.txt', "a")

    img_itr = len(os.listdir(f'{folder_name}/{images_folder}'))

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

        image_buffer = deque([], maxlen=3)

        while not solved and step < 200:
            if step < 5: 
                state, reward, done, info = env.step(np.zeros((7,), dtype=np.float32))
                image = cv2.resize(state[0], (159, 90), interpolation=cv2.INTER_AREA) 
                image_buffer.append(image)
            else:
                image = cv2.resize(state[0], (159, 90), interpolation=cv2.INTER_AREA) 
                image_buffer.append(image)
                latest_image = np.concatenate(image_buffer, axis=-1)
                state = (latest_image, state[1])
                action = agent.sample_actions(state) 
                state, reward, done, info = env.step(action) 

                if step < 5:
                    continue
                
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
                # start_0 -= change
                break
    
    agent.close()
    fl.close()


if __name__ == "__main__":
    folder_name='inference_test_data_2'
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
        
    images_folder='images_1'
    if not os.path.exists(f'{folder_name}/{images_folder}'): 
        os.makedirs(f'{folder_name}/{images_folder}')
    for i in range(8):
        fn(i, 20000, folder_name, images_folder, 1)
        fn(i, 50000, folder_name, images_folder, 1)
        
    images_folder='images_3'
    if not os.path.exists(f'{folder_name}/{images_folder}'): 
        os.makedirs(f'{folder_name}/{images_folder}')
    for i in range(8):
        fn(i, 20000, folder_name, images_folder, 3)
        fn(i, 50000, folder_name, images_folder, 3)
 