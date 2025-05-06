import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES']='0'

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


OB_TYPE_1 = "MASK"
OB_TYPE_2 = "OH"
OB_TYPE_3 = "3d_position"
OB_TYPE_4 = 'clip'

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--env_name', default='isaac_create', type=str)
    parser.add_argument('--image_height', default=90, type=int)     # Mode: img, img_prop
    parser.add_argument('--image_width', default=160, type=int)      # Mode: img, img_prop     
    parser.add_argument('--image_history', default=3, type=int)     # Mode: img, img_prop
    parser.add_argument('--ob_type', default=OB_TYPE_4, type=str)
    
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=300_000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=5_000, type=int)
    parser.add_argument('--env_steps', default=300_000, type=int)
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
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int)
    parser.add_argument('--num_eval_episodes', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=10_000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=500_000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)

    parser.add_argument('--img_aug_path', default='', type=str)
    parser.add_argument('--buffer_save_path', default='', type=str) # ./buffers/
    parser.add_argument('--buffer_load_path', default='', type=str) # ./buffers/

    args = parser.parse_args()
    return args

def main(seed=-1):
    import time
    import shutil
    import numpy as np
    from jsac.helpers.logger import Logger
    from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
    from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv

    task_start_time = time.time()
    args = parse_args()

    if seed != -1:
        args.seed = seed

    if not args.sync_mode:
        assert args.mode != MODE.PROP, "Async mode is not supported for proprioception only tasks." 
    
    args.start_episode, args.start_step = 0, 0

    sync_mode = 'sync' if args.sync_mode else 'async'
    args.name = f'{args.env_name}_{args.ob_type}_Dist'

    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'
    print(args.work_dir)

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
            print('Previous work dir removed.')
        elif inp == '':
            pass
        else:
            exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    if args.save_model:
        make_dir(args.model_dir)
        
    args.net_params = config

    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb, wandb_project_name, 
                   wandb_run_name, args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)

    set_seed_everywhere(seed=args.seed)
    
    env = WrappedEnv(env_c, episode_max_steps=200, start_step=0, start_episode=0)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    
    print('Image shape: ',  args.image_shape, ',  proprioception_shape: ', args.proprioception_shape)

    sync_queues = None
    if args.sync_mode:
        agent = SACRADAgent(vars(args)) 
    else:
        sync_queues = (mp.Queue(), mp.Queue())
        agent = AsyncSACRADAgent(vars(args), sync_queues)

    update_paused = True
    state = env.reset()
    first_step = True

    while env.total_steps < args.env_steps:
        t1 = time.time()
        if env.total_steps < args.init_steps:
            action = (np.random.random(args.action_shape) * 2) - 1
        else:
            action = agent.sample_actions(state)
        t2 = time.time()
        next_state, reward, done, info = env.step(action)

        # if env.total_steps > args.init_steps:
        # time.sleep(0.2)
        t3 = time.time()

        mask = 1.0 if not done or 'truncated' in info else 0.0
        
        agent.add(state, action, reward, next_state, mask, first_step)
        first_step = False
        state = next_state

        if done or 'truncated' in info:
            state = env.reset()
            first_step = True
            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - task_start_time
            info['dump'] = True
            L.push(info)

        if env.total_steps >= args.init_steps and env.total_steps % args.update_every == 0:
            if not args.sync_mode and update_paused: 
                agent.resume_update()
            if not update_paused and sync_queues:
                sync_queues[1].get(timeout=300)
            if sync_queues:
                sync_queues[0].put(1)
                update_paused = False
            update_infos = agent.update()
            if update_infos is not None and env.total_steps % args.log_every == 0:
                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False

                    L.push(update_info)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)

    if not args.sync_mode:
        agent.pause_update()
    if args.save_model:
        agent.checkpoint(env.total_steps)
        
    L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    
    from jsac.envs.isaac_create_reacher.create_env import CreateReacherEnv
    env_c = CreateReacherEnv('RLC/JSAC/jsac/envs/isaac_create_reacher/create_arena.usd', 
                        headless=True, 
                        image_width=160, 
                        image_height=90,
                        ob_type=OB_TYPE_4,
                        randomize_target_pos=True)
    
    for i in range(6, 11):
        main(seed=i)
