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


# For Detic
```bash

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.4.1cu121

cd Detic && pip install -r requirements.txt

wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O Detic/models/ViT-B-32.pt

wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

# For Owlv2
```bash
Download it from https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/main inside owlv2 folder
```