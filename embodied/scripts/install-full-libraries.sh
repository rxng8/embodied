#!/bin/bash -l

# Basically install what is inside pyproject.toml dependencies but with pytorch

# Torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Jax optimiation libraries and tf-cpu
pip install optax==0.2.2 flax==0.8.2 tensorflow-cpu==2.16.1 tf-keras==2.16.0 tensorflow-probability==0.24.0
# JAX
pip install -U "jax[cuda12]"
# Related plotting libraries
pip install seaborn rich ruamel.yaml==0.17.32 opencv-python opencv-python-headless pettingzoo==1.24.3 imageio==2.33.1 tensorflow-datasets
# RL Benchmark lib
pip install gymnasium==0.29.1 gym==0.24.1 dm-control
# NLP
pip install tokenizers==0.19.1
pip install 'transformers[torch]==4.41.1'
# Related ml libraris
pip install einops