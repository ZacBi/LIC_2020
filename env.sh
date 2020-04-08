# Add mirrors source
echo "********** Add mirrors source **********"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

# Custom channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --show channels

# Create new env
echo "********** Create conda env 'torch' **********"
conda create -n torch python=3.7 pytorch=1.4.0 cudatoolkit=9.2 transformers torchtext ipython --yes