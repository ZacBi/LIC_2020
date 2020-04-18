# Add mirrors source
echo "********** Add mirrors source **********"
conda config --add envs_dirs ~/work/conda/env/
conda config --add pkgs_dirs ~/work/conda/pkgs/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --remove channels conda-forge
conda config --remove channels defaults
conda config --show channels

# Create new env
# TODO: use if else statement to check whether env has existed
# echo "********** Create conda env 'torch' **********"
# conda create -n torch python=3.7 --yes