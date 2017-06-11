# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tune momentum as well as learning rate using local quadratic approximation.

The implmentation here can be a drop-in replacement for any optimizer in Tensorflow. It supports both ```minimize``` and ```apply_gradients``` function like any tensorflow optimizer. For more technical details, please refer to our paper
[YellowFin and the Art of Momentum Tuning](TODO).

## Setup instructions for experiments
Please clone the master branch and follow the instructions to run YellowFin on ResNet for CIFAR10, Bottleneck Resnet on CIRAR100 and LSTM on Penn Treebank. The models we use are slightly adapted from Tensorflow [ResNet](https://github.com/tensorflow/models/tree/master/resnet) and [LSTM](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).

### download data
Please use the data/download.sh script to download CIFAR10/100 and Penn Treebank dataset. It may take a few minutes depending on the network condition. Other datasets are self-included in the repo.
```
cd data
bash download.sh
```

### run CIFAR10/100 ResNets experiments
The experiments on 110 layer ResNet with CIFAR10 and 164 layer ResNet with CIFAR100 can be launched using
```
cd cifar/scripts
python CIFAR10-release.py (for CIFAR10)
python CIFAR100-release.py (for CIFAR10)
```

### run Penn Treebank LSTM experiments
The experiments on multiple-layer LSTM on Penn Treebank can be launched using
```
cd ptb/scripts
python PTB-release.py
```

