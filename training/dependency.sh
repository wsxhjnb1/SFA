# Update the repositories
sudo apt-get update 

# Install packages
sudo apt-get install -y build-essential cmake curl ca-certificates sudo less htop git tzdata wget tmux zip unzip zsh stow subversion fasd

sudo apt-get install -y python3-pip

# General packages
pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops zstandard psutil h5py twine gdown

# After installing spacy, download the English model
python -m spacy download en_core_web_sm

# Install hydra and its plugins
pip install hydra-core==1.3.1 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 pyrootutils rich

# Core deep learning and NLP packages
pip install transformers==4.25.1 datasets==2.8.0 pytorch-lightning==1.8.6 triton==2.0.0.dev20221202 wandb==0.13.7 timm==0.6.12 torchmetrics==0.10.3

# For MLPerf
pip install git+https://github.com/mlcommons/logging.git@2.1.0

# FlashAttention and its CUDA extensions
pip install flash-attn==2.2.1

# Note: Installing CUDA extensions usually requires appropriate NVIDIA tools and libraries to be installed, so it might be a bit more involved than a simple pip install.
# Clone the repository
git clone https://github.com/HazyResearch/flash-attention

# Checkout the specified version
cd flash-attention
git checkout v2.2.1

# Install the extensions
pip install ./csrc/fused_softmax
pip install ./csrc/rotary
pip install ./csrc/xentropy
pip install ./csrc/layer_norm
pip install ./csrc/fused_dense_lib
pip install ./csrc/ft_attention

# Clean up by removing the cloned repository
cd ..
rm -rf flash-attention

git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 23.05
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" . && cd .. && rm -rf apex

pip install transformers --upgrade
pip install sentencepiece

export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "openwebtext"

python run.py experiment=owt/gpt2s-flash trainer.devices=1  # 125M
# python run.py experiment=owt/gpt2m-flash trainer.devices=8  # 355M
# python run.py experiment=owt/gpt2l-flash trainer.devices=8  # 760M
# python run.py experiment=owt/gpt2xl-flash trainer.devices=8  # 1.6B

# export PYTHONPATH=$PWD:$PYTHONPATH
# pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "pile"

# python run.py experiment=pile/gpt3s-flash trainer.devices=8  # 125M
# python run.py experiment=pile/gpt3m-flash trainer.devices=8  # 355M
# python run.py experiment=pile/gpt3l-flash trainer.devices=8  # 760M
# python run.py experiment=pile/gpt3xl-flash trainer.devices=8  # 1.3B
# python run.py experiment=pile/gpt3-2.7B-flash-hdim128 trainer.devices=8  # 2.7B

# python setup.py install