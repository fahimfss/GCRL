import gymnasium as gym

from stable_baselines3 import A2C
import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

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
parser.add_argument('--condition_type', default='mask', type=str) # "mask", "object_image", "one_hot" 
parser.add_argument('--mask_type', default='ground_truth', type=str)  # "ground_truth", "gdino_sync", "gdino_async", "gt_gdino_async"
parser.add_argument('--mask_delay_type', default='none', type=str)
parser.add_argument('--mask_delay_steps', default=2, type=int) 
parser.add_argument('--reward_mode', default='distance', type=str)
parser.add_argument('--step_time', default=0.05, type=float)

parser.add_argument('--max_time_steps', default=250, type=int)



parser.add_argument('--logdir', default='/home/hany606/scratch/rlc_ppo_results/', type=str)

args = parser.parse_args()


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


env_name = 'FrankaEnv-v0'

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
                   condition_type=args.condition_type,
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

def cb(x):
    print(x)
    return x % 250


env = DummyVecEnv([make_env(env_name, i, seed=0) for i in range(2)])
env.render_mode = 'rgb_array'

envs = VecVideoRecorder(env, "/home/hany606/repos/RLC/training/tmp", record_video_trigger=cb, video_length=250)
envs = VecMonitor(env)
envs = VecFrameStack(envs, n_stack = 3)

print(env.observation_space)

# model = A2C("MultiInputPolicy", env, verbose=1)
model = PPO(CustomMultiInputPolicy, envs, n_steps = 1024, batch_size = 64, verbose=0,)

model.learn(total_timesteps=1000)

vec_env = model.get_env()
obs, info = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)