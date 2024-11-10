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
  
cd ../training

# Start training using: 
python3 task_rlc_0.py
```
