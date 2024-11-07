#!/bin/bash 
#SBATCH --account=rrg-ashique
#SBATCH --cpus-per-task=5
#SBATCH --array=2,4,5,9
#SBATCH --time=0-15:00
#SBATCH --mem=42G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=fshahri1@ualberta.ca
#SBATCH --mail-type=ALL 

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/fshahri1/projects/def-ashique/fshahri1/jsac_rlc/JRLCENV/bin/activate
  
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 

sleep 5

python3 /home/fshahri1/projects/def-ashique/fshahri1/jsac_rlc/RL-Chemist/task_rlc.py --seed=$SLURM_ARRAY_TASK_ID 
 
