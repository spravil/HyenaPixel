# Semantic Segmentation on ADE20k

We tested the segmentation capabilites of your backbone network with the help of the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) framework.
To setup the environment run `bash setup.sh`.
The script will setup the framework and also downloads the [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset. 
Once the setup is finished the training can be started with `sbatch upernet_hpx_former_s18.sh` using Slurm.
