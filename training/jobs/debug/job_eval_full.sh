#!/bin/bash 
#SBATCH --account=def-ashique
#SBATCH --cpus-per-task=4
#SBATCH --array=2
#SBATCH --time=0-10:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline

module load StdEnv/2023 gcc opencv cuda/12.2 python/3.10 mujoco/3.1.6
source /home/hany606/envs/RLCENV/bin/activate

cd /home/hany606/repos/RLC/training

env_mode="eval"
# reward_mode="distance"
mask_type="ground_truth"
# condition_type="mask"
env_name="FrankaEnv-debug-v0"
eval_env_name=$env_name

export PYTHONBREAKPOINT="pudb.set_trace"
export MUJOCO_GL="egl" 
export PYOPENGL_PLATFORM="egl" 
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export PYTHONPATH=$PYTHONPATH:/home/hany606/repos/RLC/mj_envs
export WANDB_MODE=offline 

for env_mode in "eval" "eval_ofd"; do
    for reward_mode in "distance" "sparse" "mask_size"; do
        for condition_type in "mask" "one_hot" "object_image"; do
            export WANDB_DIR=/home/hany606/scratch/eval_rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name-$env_mode/
            python eval_id_ppo.py --env_name=$env_name \
                                --num_envs=4 \
                                --group=ppo-sb3-$reward_mode-$mask_type-$condition_type-$env_name-$env_mode \
                                --logdir=$WANDB_DIR \
                                --reward_mode=$reward_mode \
                                --mask_type=$mask_type \
                                --condition_type=$condition_type \
                                --models_dir=/home/hany606/scratch/rlc_ppo_sb3-$reward_mode-$mask_type-$condition_type-$env_name/$env_name/ \
                                --eval_env_name=$eval_env_name \
                                --save_video \
                                --env_mode=$env_mode || {
                echo "Error occurred with reward_mode=$reward_mode and condition_type=$condition_type and env_mode=$env_mode. Skipping to next iteration."
                continue
            }
        done
    done
done
