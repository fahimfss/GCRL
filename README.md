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


# GroundingDINO
```bash
cd GroundingDINO/asset

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

```
For GroundingDINO setup with offline jobs, check this [issue](https://github.com/IDEA-Research/GroundingDINO/issues/218)

Change in `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` the following line: `text_encoder_type = "/home/hany606/.cache/huggingface/hub/models--bert-base-uncased"`

Change in `GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py` the following line: `text_encoder_type = "/home/hany606/.cache/huggingface/hub/models--bert-base-uncased"`
