# Online Outlier Detection in Open Feature Spaces (OODOFS)



## Setting up the environment

* __Python version:__ `python 3.9.7`
* __Torch version:__ `Pytorch 1.12.1`
* __Cuda version:__ `CUDA 11.6`

* __Dependencies:__ To install the dependencies using conda, please follow the steps below:

		conda create --name ood python=3.9.7
        conda activate ood
		conda install pytorch==1.12.1 torchvision cudatoolkit=11.6 -c pytorch 

## Running Experiments

Running the main experiments
```
python train.py
```

