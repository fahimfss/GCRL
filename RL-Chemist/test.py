# import gymnasium as gym
# import time

# env_name = "UR10eMask4C-v1"

# env = gym.make(f'mj_envs.robohive.envs:{env_name}', eval_mode=False)

# state = env.reset()
# print(state['image'].shape, type(state['image']), state['image'].dtype)
# print(state['vector'].dtype, state['vector'].shape)

# print(env.observation_space)
# print(env.action_space.shape[0])
# print(state['vector'].shape[0])
 

# times = []

# for i in range(200):
#     t1 = time.time()
#     obs = env.step(env.action_space.sample())
#     t2 = time.time()
    
#     times.append(t2-t1)
    
#     print(obs[0]['image'].shape)
#     # print(obs[0]['vector'].shape)
    
# avgt = sum(times) / len(times)
# print(avgt * 1000)


import gymnasium as gym
from jsac.envs.robohive_envs.ur104C import UR10_4C_ENV
from jsac.helpers.utils import WrappedEnv

env_name = "UR10eMask4C-v1"
env = WrappedEnv(UR10_4C_ENV(env_name), 250)


env.reset()

for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())
    
    if "truncated" in info: 
        if i < 300:
            env.reset(create_vid=True)
        else: 
            env.reset()
        
env.reset()
env.close()