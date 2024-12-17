import gymnasium as gym

from stable_baselines3 import A2C
import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


class CustomDictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):  # Adjust features_dim if needed
        super(CustomDictFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),  # Adjust padding to fit your needs
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

def make_env(env_name, idx, seed=0, eval_mode=False):
    def _init():
        from collections import OrderedDict
        # We are using a single frame
        env_kwargs = OrderedDict(
                    image_width=212, 
                    image_height=120,
                    # image_history=args.image_history,  # this is useless here as I am not using the UnifiedEnv class
                    # mask_delay_type=args.mask_delay_type, # this is useless here 
                    # mask_delay_steps=args.mask_delay_steps, # this is useless here 
        )
        env = gym.make(f'robohive.envs:{env_name}', eval_mode=eval_mode, **env_kwargs)
        env.seed(seed + idx)
        env = WrapperReset(env)
        env = TimeLimitWrapper(env, max_time_steps=200)
        return env
    return _init

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

# model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs, info = vec_env.reset()
for i in range(1000):
    action = vec_env.action_space.sample()
    obs, reward, done, info = vec_env.step(action)
