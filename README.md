# LIC_2020

For Language and Intelligence Chanllenge task 5: [Event Extraction](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true&lang=en)

## Pipeline

1. Create your branch and develop in it.  
2. Discussion and code review before merge.

## Discussion & Reference

Discussion: [issue2](https://github.com/ZacBi/LIC_2020/issues/2)
Reference: [issue3](https://github.com/ZacBi/LIC_2020/issues/3)

## FYI

1. Think about using **continous integration** while developing(github->action)  
2. Consider python package (wheel or .egg) for using some common utils(F-measure or others), For more info: [link](www.pythonwheels.com/)  
3. Use [Click](https://click.palletsprojects.com/en/7.x/) instead of python build-in Argparse for command develop.

------

## Pipeline in Paddle env

```sh
# Clone repo
git clone -b ner_crf https://github.com/ZacBi/LIC_2020.git
cd ./LIC_2020

# Conda env
sh env.sh
source activate torch

# Install package (if your need)
conda install --yes --file requirements_conda.txt
pip install -e ./ --no-binary :all:
```
