cd ~/work
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./

echo "********** Install additional packages **********"
conda install --file --yes requirements.txt