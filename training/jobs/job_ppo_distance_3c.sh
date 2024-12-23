#!/bin/bash 
#SBATCH --account=def-ashique
#SBATCH --cpus-per-task=4
#SBATCH --array=2,3,4,5
#SBATCH --time=0-23:30
#SBATCH --mem=66G
#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/hany606/envs/RLCENV/bin/activate

cd /home/hany606/repos/RLC/training

reward_mode="distance"
mask_type="object_image"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type/
export PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs
export WANDB_MODE=offline




python task_rlc_0_ppo.py --env_name=FrankaEnv-v0 --num_envs=4 --group=ppo-sb3-$reward_mode-$mask_type --logdir=$WANDB_DIR --seed=$SLURM_ARRAY_TASK_ID
