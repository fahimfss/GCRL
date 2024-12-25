# import gym
import gymnasium as gym

import os
from gymnasium import spaces
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch 
import random
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv
#import mujoco_py
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime
import time
from wandb.integration.sb3 import WandbCallback

import numpy as np
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
# parser.add_argument('--mask_delay_type', default='none', type=str)
# parser.add_argument('--mask_delay_steps', default=1, type=int) 
parser.add_argument('--mask_type', default='ground_truth', type=str)  # "ground_truth", "gdino_sync", "gdino_async", "gt_gdino_async"
parser.add_argument('--mask_delay_type', default='none', type=str)
parser.add_argument('--mask_delay_steps', default=2, type=int) 
parser.add_argument('--reward_mode', default='distance', type=str)
parser.add_argument('--step_time', default=0.05, type=float) 


parser.add_argument('--max_time_steps', default=250, type=int)



parser.add_argument('--logdir', default='/home/hany606/scratch/rlc_ppo_results/', type=str)



args = parser.parse_args()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        obs_vecs = self.training_env.get_attr('get_obs_vec')  # Returns a list of observations from all environments
        average_obs = np.mean([np.mean(obs) for obs in obs_vecs], axis=0)  # Compute the average observation
        self.logger.record("average_obs", average_obs)
        return True

class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for feature concatenation
        )
        # Vector processing network
        self.mlp = nn.Linear(observation_space.spaces['vector'].shape[0], 14)
        
        # Calculate the total concatenated feature dimension
        self._features_dim = 24974 #observation_space.spaces['image'].shape[0]**2 + 16  # Adjust based on actual output dimensions of cnn and mlp

    def forward(self, observations):
        image = observations['image']#.permute(0, 3, 1, 2) # already permuted
        # self.cnn(BxCxHxW)
        image_features = self.cnn(image)
        vector_features = self.mlp(observations['vector'])
        concatenated_features = torch.cat([image_features, vector_features], dim=1)
        return concatenated_features

class CustomMultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMultiInputPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomDictFeaturesExtractor,
                                                     features_extractor_kwargs={},
                                                     net_arch=[{'vf': [512, 512], 'pi': [512, 512]}])  # Adjust architecture if needed


class WrapperJSAC(gym.Wrapper):
    def __init__(self, env):
        super(WrapperJSAC, self).__init__(env)
        self.env = env
        self.counter = 0
        self.observation_space = gym.spaces.Dict({
            'image': env.image_space,
            'vector': env.proprioception_space,
        })
        

    def reset(self, *args, **kwargs):
        self.counter = 0
        obs = self.env.reset(*args, **kwargs)
        return {'image': obs[0], 'vector': obs[1]}
    
    def step(self, *args, **kwargs):
        self.counter += 1
        obs, r, done, info = self.env.step(*args, **kwargs)
        return {'image': obs[0], 'vector': obs[1]}, r, done, False, info

class WrapperReset(gym.Wrapper):
    def __init__(self, env):
        super(WrapperReset, self).__init__(env)
        self.env = env
        self.counter = 0

    def reset(self, *args, **kwargs):
        self.counter = 0
        obs = self.env.reset(*args, **kwargs)
        return obs, {}
    
    def step(self, *args, **kwargs):
        self.counter += 1
        return self.env.step(*args, **kwargs)

class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_time_steps=200):
        super(TimeLimitWrapper, self).__init__(env)
        self.env = env
        self.max_time_steps = max_time_steps
        self.current_time_step = 0
    
    def reset(self, *args, **kwargs):
        self.current_time_step = 0
        return self.env.reset(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        self.current_time_step += 1
        obs, reward, done, truncated, info = self.env.step(*args, **kwargs)
        if self.current_time_step >= self.max_time_steps:
            truncated = True
        return obs, reward, done, truncated, info

        
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env_name, idx, seed=0, eval_mode=False):
    def _init():
        from collections import OrderedDict
        # We are using a single frame
        env_kwargs = OrderedDict(
                    image_width=args.image_width, 
                    image_height=args.image_height,
                    # image_history=args.image_history,  # this is useless here as I am not using the UnifiedEnv class
                    # mask_delay_type=args.mask_delay_type, # this is useless here 
                    # mask_delay_steps=args.mask_delay_steps, # this is useless here 
        )
        env = gym.make(f'robohive.envs:{env_name}', eval_mode=eval_mode, **env_kwargs)
        env.seed(seed + idx)
        env = WrapperReset(env)
        env = TimeLimitWrapper(env, max_time_steps=args.max_time_steps)
        return env
    # return _init
    def _init_jsac():
        env_mode = 'train' if not eval_mode else 'eval'
        step_time = None
        if args.step_time > 0:
            step_time = args.step_time

        from jsac.envs.rl_chemist.env import RLC_Env
        # We are using a single frame
        env = RLC_Env(args.env_name, 
                   args.image_history, 
                   args.image_width, 
                   args.image_height,
                   mask_type=args.mask_type, # "ground_truth", "gdino_sync", "gdino_async"
                   mask_delay_type=args.mask_delay_type, # "none", "n_step", "sequential"
                   mask_delay_steps=args.mask_delay_steps,
                   reward_mode=args.reward_mode, # "distance", "mask_size"
                   env_mode=env_mode, # "train", "eval_ofd", "eval", "inference_1", "inference_3"
                   step_time=step_time
                   )
        env = WrapperJSAC(env)
        env = WrapperReset(env)
        env = TimeLimitWrapper(env, max_time_steps=args.max_time_steps)
        return env
    return _init_jsac


def main():
    training_steps = 3500000
    env_name = args.env_name
    start_time = time.time()
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ENTROPY = 0.01
    LR = linear_schedule(args.learning_rate)
    CR = linear_schedule(args.clip_range)

    time_now = time_now + str(args.seed)
    log_path = os.path.join(args.logdir, env_name, time_now, 'policy_best_model')
    tensorboard_log_path = os.path.join(args.logdir, env_name, time_now, 'tensorboard')
    video_log_path = os.path.join(args.logdir, env_name, time_now, 'videos')
    # wandb_logdir = os.path.join(args.logdir)
    
    IS_WnB_enabled = True
    loaded_model = time_now #'2024_09_25_13_42_113'

    set_seed_everywhere(seed=args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    num_cpu = args.num_envs

    env = DummyVecEnv([make_env(env_name, i, seed=args.seed) for i in range(num_cpu)])
    env.render_mode = 'rgb_array'

    envs = VecVideoRecorder(env, video_log_path, record_video_trigger=lambda x: x % 2000 == 0, video_length=250)
    envs = VecMonitor(env)
    envs = VecFrameStack(envs, n_stack = 3)

    ## EVAL
    eval_env = DummyVecEnv([make_env(env_name, i, seed=args.seed, eval_mode=True) for i in range(1)])
    eval_env.render_mode = 'rgb_array'
    eval_envs = VecFrameStack(eval_env, n_stack = 3)
    
    eval_callback = EvalCallback(eval_envs, best_model_save_path=log_path, log_path=log_path, eval_freq=2000, n_eval_episodes=20, deterministic=True, render=False)
    
    print('Begin training')
    print(time_now)


    # Create a model using the vectorized environment
    #model = SAC("MultiInputPolicy", envs, buffer_size=1000, verbose=0)
    model = PPO(CustomMultiInputPolicy, envs, ent_coef=ENTROPY, learning_rate=LR, clip_range=CR, n_steps = 1024, batch_size = 64, verbose=0, tensorboard_log=tensorboard_log_path)
    #model = PPO.load(r"./Reach_Target_vel/policy_best_model/" + env_name + '/' + loaded_model + '/best_model', envs, verbose=1, tensorboard_log=f"runs/{time_now}")
    
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        config = {
            "policy_type": 'PPO',
            'name': time_now,
            "total_timesteps": training_steps,
            "env_name": env_name,
            "dense_units": 512,
            "activation": "relu",
            "max_episode_steps": 250,
            "seed": args.seed,
            "entropy": ENTROPY,
            "lr": args.learning_rate,
            "CR": args.clip_range,
            "num_envs": args.num_envs,
            "loaded_model": loaded_model,
            "log_path": log_path,
            "tensorboard_log_path": tensorboard_log_path,
            "video_log_path": video_log_path,
            "reward_mode": args.reward_mode,
            "mask_type": args.mask_type,
            "mask_delay_steps": args.mask_delay_steps,
            "mask_delay_type": args.mask_delay_type,
            "image_history": args.image_history,
            "step_time": args.step_time,
            "max_time_steps": args.max_time_steps,
            "logdir": args.logdir
            }
        #config = {**config, **envs.rwd_keys_wt}
        run = wandb.init(project="RL-Chemist_Reach",
                        group=args.group,
                        settings=wandb.Settings(start_method="fork"),
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        monitor_gym=True,  # auto-upload the videos of agents playing the game
                        save_code=True,  # optional
                        entity='hanyhamed606',
                        tags=[
                            f"HWC={args.image_height}x{args.image_width}x3"
                            f"reward_mode={args.reward_mode}",
                            f"mask_type={args.mask_type}",
                            f"mask_delay_steps={args.mask_delay_steps}",
                            f"mask_delay_type={args.mask_delay_type}",
                            ],
                        # dir=wandb_logdir,
                        )
        wandb.run.name = '-'.join([env_name, time_now, str(args.seed)])
        # wandb.tensorboard.patch(root_logdir=tensorboard_log_path)

    except ImportError as e:
        IS_WnB_enabled = False
        print(e)
        pass 

    callbacks = []
    callbacks += [eval_callback, WandbCallback(gradient_save_freq=100)]
    
    callback = CallbackList(callbacks)
    
    model.learn(total_timesteps=training_steps, callback=callback)# , tb_log_name=env_name + "_" + time_now)

    if IS_WnB_enabled:
        run.finish()

if __name__ == "__main__":
    # TRAIN
    main()