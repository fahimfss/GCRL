#!/bin/bash 
#SBATCH --account=def-ashique
#SBATCH --cpus-per-task=4
#SBATCH --array=2,3,4,5
#SBATCH --time=0-40:30
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/hany606/envs/RLCENV/bin/activate

cd /home/hany606/repos/RLC/training

reward_mode="mask_size"
mask_type="ground_truth"
condition_type="mask"
env_name="FrankaEnv-debug-v0"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/
export PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs
export WANDB_MODE=offline




python task_rlc_0_ppo.py --env_name=$env_name --num_envs=4 --group=ppo-sb3-$reward_mode-$mask_type-$condition_type-$env_name --logdir=$WANDB_DIR --seed=$SLURM_ARRAY_TASK_ID  --reward_mode=$reward_mode --mask_type=$mask_type --condition_type=$condition_type
