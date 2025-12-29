## Overview
This repository contains the associated code for the paper General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks [[ArXiv]](https://arxiv.org/abs/2510.06277). In this paper, we introduce a mask-based goal representation for visual GCRL that enables efficient learning, strong generalization to unseen objects, and effective sim-to-real transfer without requiring 3D positional information.


Video Demo: 

https://github.com/user-attachments/assets/924baef0-f0bf-4a0d-861d-7d86b07ced9e

### Experiment Highlight: Pick-up from Scratch
We trained an end-to-end RL agent capable of picking up objects of interest using images from a wrist-mounted camera and without any 3D positional data, learning from scratch.  

<img width="2452" height="436" alt="image" src="https://github.com/user-attachments/assets/d0807b21-df34-4b13-a103-087772548a23" />

GCRL agents trained using our proposed method (mask-based GCRL and reward system) significantly outperformed other agents, such as 3D position-based GCRL and distance-based reward systems, in post-training trials.  

<img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/c648d030-9ed2-40d6-ad35-4405bae7043c" />


## Installation

```
# Create a conda env and install JSAC
conda create -n rlc python=3.10
conda activate rlc

pip install -U "jax[cuda12]==0.4.30"

cd JSAC
pip install -e .

# Install Robohive
cd ../mj_envs
pip install -e .
  
# [Optional] Install GroundingDINO and DETIC
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+2a420edpt2.4.1cu121
cd Detic && pip install -r requirements.txt
cd GroundingDINO && pip install -r requirements.txt
transformers-cli download bert-base-uncased
```

### Running the Experiments
#### Experiment 1 and 2 (Mujoco)
```python3 training/task_rlc_gt.py --seed=0 --env_name=UR10eEnv-v1 --goal_type=G1_Mask --reward_mode=mask_size```

Options:  
env_name = 'FrankaEnv-v1' or 'UR10eEnv-v1'  
goal_type = 'G1_Mask', 'G2_OH', 'G3_3d', 'G4_Clip' or 'G5_TS'  
reward_mode = 'distance' or 'mask_size'  

#### Experiment 1 (Isaac Sim)
Download [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html) (version 5.1.0 as of Dec 2025) and install JSAC using the Isaac Sim's Python interpreter. After that, run the code using:
```python3 training/task_isaac_async.py```

#### Experiment 3
```python3 training/task_rlc_gt.py --seed=0 --env_name=UR10eEnv-v2 --goal_type=G1_Mask --reward_mode=mask_size```

Options:  
env_name = 'FrankaEnv-v2' or 'UR10eEnv-v2'

#### Real-world Learning from Scratch using Franka
Real-world training requires a Franka Emika robot and the [franka_ros_interface](https://github.com/justagist/franka_ros_interface) repository installed. To start training, run the following code:
```python3 training/task_rlc_franka_rw.py```
