# Object Detection on MS-COCO

We tested the detection capabilites of your backbone network with the help of the [mmdetection](https://github.com/open-mmlab/mmdetection) framework.
First download the [MS-COCO](https://cocodataset.org) Dataset.
To setup the environment run `bash setup.sh /path/to/coco`.
Once the setup is finished the training can be started with `sbatch cascade_mask_rcnn_hpx_former_s18.sh` using Slurm.
