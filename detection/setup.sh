#!/bin/bash
COCO=$1

# Check arguments
if [[ -z $COCO ]];
then
    echo "Missing mandatory arguments: dataset."
    echo "Usage: ./setup.sh [dataset]"
    exit 1
fi

# Check if folder exists
if [ ! -d "$COCO" ]; then
  echo "Directory $COCO does not exist."
  exit 1
fi

# Install requirements
pip install -U openmim
mim install "mmengine==0.10.1"
mim install "mmcv==2.1.0"

# Clone and install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
( cd mmdetection || exit; git checkout fe3f809 )
( cd mmdetection || exit; pip install -e . )

# Test installation
python test_installation.py

# Link dataset
mkdir data
( cd data || exit; ln -s "$COCO" . )

