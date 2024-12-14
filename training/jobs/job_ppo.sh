#!/bin/bash 
#SBATCH --account=def-ashique
#SBATCH --cpus-per-task=4
#SBATCH --array=2,3
#SBATCH --time=0-23:30
#SBATCH --mem=66G
#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/hany606/envs/RLCENV/bin/activate

cd /home/hany606/repos/RLC/training


export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR=/home/hany606/scratch/wandb_rlc/
export PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs
export WANDB_MODE=offline

python task_rlc_0_ppo.py --env_name=FrankaEnv-v0 --num_envs=4 --group=ppo_sb3 --seed=$SLURM_ARRAY_TASK_ID
