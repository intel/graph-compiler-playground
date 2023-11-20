conda create --name dl python=3.11 -y
conda activate dl
conda install pytorch torchvision  cpuonly -c pytorch

# git submodule init
# git submodule update

# itex
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch