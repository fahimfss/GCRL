#!/bin/bash 
env_mode="eval"
reward_mode="distance"
mask_type="ground_truth"
condition_type="mask"
env_name="FrankaEnv-debug-v0"
# env_name="UR10eEnv-v0"
eval_env_name=$env_name


# MUJOCO_GL="egl" PYOPENGL_PLATFORM="egl" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DIR=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/ PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs WANDB_MODE=offline python task_rlc_0_ppo.py --env_name=$env_name --num_envs=4 --group=ppo-sb3-$reward_mode-$mask_type-$condition_type-$env_name --logdir=$WANDB_DIR --seed=$SLURM_ARRAY_TASK_ID  --reward_mode=$reward_mode --mask_type=$mask_type --condition_type=$condition_type
# PYTHONBREAKPOINT="pudb.set_trace" MUJOCO_GL="egl" PYOPENGL_PLATFORM="egl" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DIR=/home/hany606/scratch/eval_rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/ PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs WANDB_MODE=offline python eval_id_ppo.py --env_name=$env_name --num_envs=4 --group=ppo-sb3-$reward_mode-$mask_type-$condition_type-$env_name --logdir=$WANDB_DIR  --reward_mode=$reward_mode --mask_type=$mask_type --condition_type=$condition_type --models_dir=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/$env_name/ --eval_env_name=$eval_env_name  --max_time_steps=50
PYTHONBREAKPOINT="pudb.set_trace" MUJOCO_GL="egl" PYOPENGL_PLATFORM="egl" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_DIR=./results_tmp/eval_rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name-$env_mode/ PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs WANDB_MODE=offline python eval_id_ppo.py --env_name=$env_name --num_envs=4 --group=ppo-sb3-$reward_mode-$mask_type-$condition_type-$env_name-$env_mode --logdir=$WANDB_DIR  --reward_mode=$reward_mode --mask_type=$mask_type --condition_type=$condition_type --models_dir=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/$env_name/ --eval_env_name=$eval_env_name  --max_time_steps=50 --save_video --env_mode=$env_mode