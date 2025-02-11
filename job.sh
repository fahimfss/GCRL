#!/bin/bash

# Specify the partition of the cluster to run on (Typically TrixieMain)
#SBATCH --partition=TrixieMain
# Add your project account code using -A or --account
#SBATCH --account AI4D-CORE-148
# Specify the time allocated to the job. Max 12 hours on TrixieMain queue.
#SBATCH --time=10:08:00
# Request GPUs for the job. In this case 4 GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name=retraining_for_pickup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G        # Adjust memory request as needed
#SBATCH --mail-type=ALL   # Send email on begin, end, and fail
#SBATCH --mail-user=huiyi.wang@nrc-cnrc.gc.ca
# Print out the hostname that the jobs is running on
hostname

echo $CUDA_HOME
conda env config vars set CUDA_HOME="/home/wanghuiy/anaconda3/envs/mujoco_env"
conda env config vars set LD_LIBRARY_PATH="/home/wanghuiy/anaconda3/envs/mujoco_env/lib/"
conda env config vars set CPATH="/home/wanghuiy/anaconda3/envs/mujoco_env/include/"

# unset PIP_CONFIG_FILE; unset PYTHONPATH; 
# Activate the conda pytorch environment created in step 1
module load conda/3-24.9.0
source activate /home/wanghuiy/anaconda3/envs/mujoco_env
#export MUJOCO_GL="egl"
#export PYOPENGL_PLATFORM="egl"

# Launch our test pytorch python files
python training/task_rlc_gt.py --seed=0 --goal_type="G4_Clip"
nvidia-smi
