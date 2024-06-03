#!/bin/bash

# Setup dataset
mkdir -p data/ade
( cd data/ade || exit; wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
( cd data/ade || exit; unzip -q ADEChallengeData2016.zip)

# Install requirements
pip install -U openmim
mim install "mmengine==0.10.1"
mim install "mmcv==2.1.0"
pip install ftfy
pip install regex 

# Clone and install mmdetection
git clone https://github.com/open-mmlab/mmsegmentation.git
( cd mmsegmentation || exit; git checkout c685fe6 )
( cd mmsegmentation || exit; pip install -e . )

# Test installation
python test_installation.py
