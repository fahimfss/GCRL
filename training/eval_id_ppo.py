import gym
from gym import spaces
from stable_baselines3 import PPO
import skvideo
import skvideo.io
import numpy as np
import os
import cv2 as cv
import random
from tqdm.auto import tqdm
import torch
import mujoco
from einops import rearrange
import json
from task_rlc_0_ppo import *
from debug import *

import argparse
parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--env_name", type=str, default='N/A', help="environment name")
parser.add_argument("--group", type=str, default='testing', help="environment name")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clip range for the policy gradient update")
parser.add_argument('--task_name', default='baseline', type=str)

parser.add_argument('--image_height', default=120, type=int)     # Mode: img, img_prop
parser.add_argument('--image_width', default=212, type=int)     # Mode: img, img_prop     
# parser.add_argument('--image_history', default=3, type=int)     # Mode: img, img_prop
parser.add_argument('--image_history', default=1, type=int)     # Mode: img, img_prop
parser.add_argument('--n_stack_frames', default=3, type=int) # TODO: refactor w/ image_history
# parser.add_argument('--mask_delay_type', default='none', type=str)
# parser.add_argument('--mask_delay_steps', default=1, type=int) 
parser.add_argument('--condition_type', default='mask', type=str) # "mask", "object_image", "one_hot" 
parser.add_argument('--mask_type', default='ground_truth', type=str)  # "ground_truth", "gdino_sync", "gdino_async", "gt_gdino_async"
parser.add_argument('--mask_delay_type', default='none', type=str)
parser.add_argument('--mask_delay_steps', default=2, type=int) 
parser.add_argument('--reward_mode', default='distance', type=str)
parser.add_argument('--step_time', default=0.05, type=float)

parser.add_argument('--max_time_steps', default=250, type=int)

parser.add_argument('--logdir', default='/home/hany606/scratch/rlc_ppo_evals/', type=str)

parser.add_argument('--models_dir', default='/home/hany606/scratch/rlc_ppo_results')
parser.add_argument('--eval_env_name', type=str, default='N/A', help="environment name")
parser.add_argument('--env_mode', type=str, default='eval', help='\in [train, eval, eval_ofd]')
parser.add_argument('--save_video', action='store_true') # by default false

args = parser.parse_args()
reward_mode = args.reward_mode
if reward_mode == "distance":
    weighted_reward_keys = {
        'distance': -1.0, 
        'contact': 0.,
        'penalty': 0.1,
        'mask_size': 0.,
        'done': 5.,
    }
elif reward_mode == "sparse":
    weighted_reward_keys = {
        'distance': 0., 
        'contact': 1.,
        'penalty': 0.,
        'mask_size': 0.,
        'solved': 10,
        'done': 100.,
    }
elif reward_mode == "mask_size":
    weighted_reward_keys = {
        'distance': 0., 
        'contact': 0.,
        'penalty': 1.,
        'mask_size': 0.9,
        'done': 5.,
    }
else:
    raise NotImplementedError(f"reward_mode == {reward_mode} is not implemented")


print(reward_mode, weighted_reward_keys)

model_src_path = args.models_dir
eval_env_name = args.eval_env_name
env_mode = args.env_mode
n_stack_frames = args.n_stack_frames

n_evals = 4
n_envs = 1
_n_evals = n_evals // n_envs 
eval_env = DummyVecEnv([make_env(args, eval_env_name, i, seed=args.seed, env_mode=env_mode) for i in range(n_envs)])
eval_env.render_mode = 'rgb_array'
eval_envs = VecFrameStack(eval_env, n_stack=args.n_stack_frames)

for d in os.listdir(model_src_path):
    model_parent_dir = os.path.join(model_src_path, d)
    if not os.path.isdir(model_parent_dir):
        continue
    model_path = os.path.join(model_parent_dir, 'policy_best_model/best_model.zip')
    model = PPO.load(model_path)
    print(f'Model loaded from {model_path}')
    mean_rewards = []
    mean_rmode_rewards = []
    frames = []
    for n in range(_n_evals):
        obs = eval_envs.reset() 
        # obs['image'] NHWC
        solved, done = np.array([False for _ in range(n_envs)]), np.array([False for _ in range(n_envs)])
        # TODO: refactor
        rewards = np.zeros((n_envs, 1)) # N, T
        rmode_rewards = np.zeros((n_envs, 1)) # N, T
        step = 0
        frames.append(rearrange(obs['image'][0], 'h w (nframes c) -> nframes h w c', nframes=n_stack_frames)) if n == 0 else None
        # TODO: add success_rate
        while step < args.max_time_steps:
            image = obs['image'].transpose(0,3,1,2)
            action, _ = model.predict({'image': image, 'vector': obs['vector']}, deterministic=True)
            obs, reward, done, info = eval_envs.step(action)
            # TODO: refactor
            rewards = np.concatenate([rewards, reward[:, None]], axis=1)
            rmode_reward = np.array([np.sum([wt*_info['rwd_dict'][key] for key, wt in weighted_reward_keys.items()], axis=0) for _info in info])
            rmode_rewards = np.concatenate([rmode_rewards, rmode_reward[:, None]], axis=1)
            step += 1
            if args.save_video:
                # take only the first image of the first vectorized environment & first evaluation episode
                _im = rearrange(obs['image'][0], 'h w (nframes c) -> nframes h w c', nframes=n_stack_frames)
                frames.append(_im) if n == 0 else None

        # TODO: refactor
        rewards = rewards[:, 1:]
        mean_rewards_per_episode = rewards.sum(1)
        mean_rewards.append(mean_rewards_per_episode.mean())

        rmode_rewards = rmode_rewards[:, 1:]
        mean_rmod_rewards_per_episode = rmode_rewards.sum(1)
        mean_rmode_rewards.append(mean_rmod_rewards_per_episode.mean())

    print(f"{np.mean(mean_rewards)}+/-{np.std(mean_rewards)}")
    print(f"{np.mean(mean_rmode_rewards)}+/-{np.std(mean_rmode_rewards)}")

    if args.save_video:
        # frames: list of HWC
        imgs = np.concatenate(frames)[:,:,:,:3]
        # TODO: Concatenate the mask
        # TODO: Add the label of the target object
        # TODO: use the function from jsac.env for saving the video instead 
        imgs = imgs[:,:,:,::-1] # BGR to RGB
        video_path = os.path.join(args.logdir, 'video')
        os.makedirs(video_path, exist_ok=True)
        skvideo.io.vwrite(os.path.join(video_path, f'{d}_video.mp4'), np.asarray(imgs), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
        print(f"Video saved in {os.path.join(video_path, f'{d}_video.mp4')}")
        # skvideo.io.vwrite('./videos'  +'/' + env_name + '/' + model_num + f'{view}_mask_video.mp4', np.asarray(frames_mask), inputdict = {'-r':'50'} , outputdict={"-pix_fmt": "yuv420p"})
    json_path = os.path.join(args.logdir, f"{d}_results.json")
    json_data = {'rew_mean': np.mean(mean_rewards), 'rew_std': np.std(mean_rewards), 'rews': mean_rewards,
                 'rmode_rew_mean': np.mean(mean_rmode_rewards), 'rmode_rew_std': np.std(mean_rmode_rewards), 'rmode_rews': mean_rmode_rewards}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


