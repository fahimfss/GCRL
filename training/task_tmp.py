import gymnasium as gym

from stable_baselines3 import A2C
import torch
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, SAC


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

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return obs, {}

env_name = 'UR10eEnv-v0' #'FrankaEnv-v0'

env = gym.make(f'robohive.envs:{env_name}', image_width = 212,
               image_height= 120, image_history=3)#, render_mode="rgb_array")
env = WrapperReset(env)

print(env.observation_space)

# model = A2C("MultiInputPolicy", env, verbose=1)
model = PPO(CustomMultiInputPolicy, env, n_steps = 1024, batch_size = 64, verbose=0,)

# model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
print(obs.keys())
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
