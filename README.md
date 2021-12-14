FastDepth
============================

## Contents
0. [Requirements](#requirements)
0. [Trained Models](#trained-models)
0. [Citation](#citation)


## Requirements
1. Install [PyTorch](https://pytorch.org/)
2. Followed by:
```
  sudo apt update && sudo apt install -y libhdf5-serial-dev hdf5-tools
  pip3 install h5py matplotlib imageio scikit-image opencv-python
  ```

## Trained Models ##
  Download the model `mobilenet-nnconv5-skipadd-pruned` from [http://datasets.lids.mit.edu/fastdepth/results/](http://datasets.lids.mit.edu/fastdepth/results/).

### Pretrained MobileNet ###

The model file for the pretrained MobileNet used in our model definition can be downloaded from [http://datasets.lids.mit.edu/fastdepth/imagenet/](http://datasets.lids.mit.edu/fastdepth/imagenet/).

## Citation
If you reference our work, please consider citing the following:

	@inproceedings{icra_2019_fastdepth,
		author      = {{Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne}},
		title       = {{FastDepth: Fast Monocular Depth Estimation on Embedded Systems}},
		booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
		year        = {{2019}}
	}
