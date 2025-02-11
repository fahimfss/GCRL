import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import flax
import numpy as np
from jax import random
import jax.numpy as jnp 
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor, get_init_data
from jsac.helpers.utils import MODE, WrappedEnv
from jsac.envs.rl_chemist.env import RLC_Env

best_actor_path = '/home/fahim/Projects/plotting/results_G1_G2/results/FrankaEnv-v1_img_prop_async_gt_G1_Mask/seed_0/eval_result/best_actor_params.pkl'

 
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

if __name__ == "__main__":
    env = RLC_Env('FrankaEnv-v1', 
                   3, 
                   120, 
                   90, 
                   mask_delay_type='none',
                   mask_delay_steps=3,
                   goal_type='G1_Mask',
                   reward_mode='distance',
                   step_time=0,
                   ofd_index=0,
                   env_mode="eval_ofd")
    
    env = WrappedEnv(env, 150)

    image_shape = env.image_space.shape 
    proprioception_shape = env.proprioception_space.shape
    action_shape = env.action_space.shape
    env_action_space = env.action_space

    rng = jax.random.PRNGKey(0)
    rng, actor = init_inference_actor(rng, 
                                        image_shape, 
                                        proprioception_shape, 
                                        config, 
                                        action_shape[-1], 
                                        False,
                                        'img_prop', 
                                        jnp.float32)
    
    rng, key1, key2 = random.split(rng, 3)
    params= actor.init(key1, key2, *get_init_data(image_shape, proprioception_shape, 'img_prop'))['params']

    with open(best_actor_path, 'rb') as f: 
        params = flax.serialization.from_bytes(params, f.read())

    episodes = 15
    for j in range(episodes): 
        state = env.reset() 
        print('s0', state[0].shape, '  s1', state[1].shape)
        for i in range(150):  
            rng, action = sample_actions(rng, 
                                        actor.apply, 
                                        params, 
                                        state, 
                                        'img_prop', 
                                        True)

            action = np.asarray(action).clip(-1, 1)
            state, reward, done, info = env.step(action) 
            if done or 'truncated' in info: 
                break