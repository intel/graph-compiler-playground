# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh -y ./Miniconda3-latest-Linux-x86_64.sh

# get github repo
conda install gh -c conda-forge --solver libmamba

# set up env
conda create -y -n ipex python=3.11
conda activate ipex
# Install ipex & pytorch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

# Install benchmarks
pip install -e .

