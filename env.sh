# Add mirrors source
echo "Add mirrors source"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pro

# Custom channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2

# Create new env
echo "Create conda env 'torch'"
conda create -n torch --file requirements_dev.txt --yes