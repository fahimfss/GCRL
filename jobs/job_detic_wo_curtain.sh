#!/bin/bash 
#SBATCH --account=def-ashique
#SBATCH --cpus-per-task=3
#SBATCH --array=[0-19]
#SBATCH --time=0-40:00
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --job-name=wo_curtain


#module load StdEnv/2023 gcc opencv cuda/12.6 python/3.10 mujoco/3.1.6
# module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
module load StdEnv/2023 gcc opencv cuda/12.6 python/3.10 mujoco/3.1.6

# source /home/fshahri1/projects/def-ashique/fshahri1/RLCENV/bin/activate
source /home/hany606/envs/rlc_tmp/bin/activate

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 /home/hany606/repos/rlc/RLC/training/task_rlc_detic.py --task_name="wo_curtain" --seed=$SLURM_ARRAY_TASK_ID 
