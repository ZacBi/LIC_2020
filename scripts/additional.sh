echo "********** Install additional packages **********"
conda install --file requirements.txt --yes
pip install transformers tensorboard seqeval
pip install -e ./ --no-binary :all: