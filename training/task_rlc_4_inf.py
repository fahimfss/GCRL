import warnings
warnings.filterwarnings("ignore")

import os
import time
import shutil
import argparse
import multiprocessing as mp
import numpy as np

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

from jsac.helpers.logger import Logger
from jsac.envs.rl_chemist.env import RLC_Env
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv

import jax
import flax
import numpy as np
from jax import random
import jax.numpy as jnp 
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor, get_init_data

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 5, 2],
        [32, 32, 5, 2],
        [32, 64, 3, 1],
        [64, 64, 3, 1],
    ],
    
    'latent_dim': 64,

    'mlp': [1024, 1024],
}

GOALTYPE_MASK = "G1_Mask"
GOALTYPE_ONE_HOT = "G2_OH"
GOALTYPE_3D = "G3_3d"
GOALTYPE_CLIP = "G4_Clip"
GOALTYPE_TARGET_STATE = "G5_TS"

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='UR10eEnv-v2', type=str)   # FrankaEnv-v1
    parser.add_argument('--task_name', default='gt', type=str)
    parser.add_argument('--goal_type', default=GOALTYPE_MASK, type=str)
    parser.add_argument('--reward_mode', default="mask_size", type=str)   # "distance", "mask_size"
    parser.add_argument('--image_height', default=90, type=int)          # Mode: img, img_prop
    parser.add_argument('--image_width', default=160, type=int)          # Mode: img, img_prop     
    parser.add_argument('--image_history', default=3, type=int)          # Mode: img, img_prop
    parser.add_argument('--step_time', default=0.0, type=float) 
    parser.add_argument('--episode_steps', default=150, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=300_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=50_000, type=int)
    parser.add_argument('--env_steps', default=1_000_000, type=int)
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
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=8, type=int)   # Sync mode: False
    
    # encoder
    parser.add_argument('--spatial_softmax', default=False, action='store_true')    # Mode: img, img_prop

    # sac
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--num_cameras', default=1, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int)
    parser.add_argument('--num_eval_episodes', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=500_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)

    parser.add_argument('--img_aug_path', default='', type=str)
    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args

def main(seed=-1, env_name=None):
    task_start_time = time.time()
    args = parse_args()

    if seed != -1:
        args.seed = seed
    
    if env_name is not None:
        args.env_name = env_name

    if not args.sync_mode:
        assert args.mode != MODE.PROP, "Async mode is not supported for proprioception only tasks." 

    args.name = f'{args.env_name}_{args.task_name}_{args.goal_type}_{args.reward_mode}_1M'
    args.work_dir += f'/results_E4 copy/{args.name}/seed_{args.seed}/'
        
    args.net_params = config

    step_time = None
    if args.step_time > 0:
        step_time = args.step_time
    env = RLC_Env(args.env_name, 
                   args.image_history, 
                   args.image_width, 
                   args.image_height, 
                   goal_type=args.goal_type,
                   reward_mode=args.reward_mode,
                   step_time=step_time,
                   ofd_index=args.seed)
    
    env = WrappedEnv(env, args.episode_steps)

    set_seed_everywhere(seed=args.seed)

    args.image_shape = env.image_space.shape 
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    
    print(f'Image shape: {args.image_shape}')
    print(f'Proprioception shape: {args.proprioception_shape}')
    print(f'Action shape: {args.action_shape}')

    rng = jax.random.PRNGKey(0)
    rng, actor = init_inference_actor(rng, 
                                      args.image_shape, 
                                      args.proprioception_shape, 
                                      config, 
                                      args.action_shape[-1], 
                                      False, 
                                      'img_prop', 
                                      jnp.float32)

    rng, key1, key2 = random.split(rng, 3)
    params = actor.init(key1, key2, *get_init_data(args.image_shape, args.proprioception_shape, 'img_prop'))['params']

    params_path = args.work_dir + 'params.pkl'
    print(params_path)

    with open(params_path, 'rb') as f: 
        params = flax.serialization.from_bytes(params, f.read())

    touch = False
    num_episods_per_object = 50

    for object_id in [0, 1, 2, 4, 5, 16]:
        stats={'object_id': object_id}
        print('\t', object_id)
        dones=[]
        for ep in range(num_episods_per_object):
            print('\t\t', ep)
            touch = False
            state = env.reset(create_vid=True, object_id=object_id)
            ret = 0
            while True:
                rng, action = sample_actions(rng, 
                                            actor.apply, 
                                            params, 
                                            state, 
                                            'img_prop', 
                                            True)

                action = np.asarray(action).clip(-1, 1)
                next_state, reward, done, info = env.step(action) 

                if reward >= 9:
                    if not touch:
                        ret += 0.5
                        touch = True
                    else:
                        ret += 0.5

                state = next_state

                if done or 'truncated' in info: 
                    dones.append(ret)
                    break
        
        stats['dones'] = dones
        with open(f'{args.work_dir}/eval_ofd_logs.txt', 'a') as f:
            f.write(str(stats) + '\n')
        
    env.close() 
    

def run(seed):
    main(seed=seed)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    seeds = [task_id, task_id + 3, task_id + 6, task_id + 9]
    seeds = [2]
    processes = []
    for s in seeds:
        p = mp.Process(target=run, args=(s,))
        p.start()
        processes.append(p)
        time.sleep(5)
    for p in processes:
        p.join()
