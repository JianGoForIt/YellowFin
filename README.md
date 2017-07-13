# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tunes momentum as well as learning rate using local quadratic approximation.

The implementation here can be **a drop-in replacement for any optimizer in Tensorflow**. It supports both ```minimize``` and ```apply_gradients``` like any tensorflow optimizer after ```from yellowfin import YFOptimizer```. 

For more technical details, please refer to our paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).

For more usage details, please refer to the inline documentation of ```tuner_utils/yellowfin.py```. Example usage can be found here for [CIFAR](https://github.com/JianGoForIt/YellowFin/blob/master/cifar/model/resnet_model.py#L160) and [PTB](https://github.com/JianGoForIt/YellowFin/blob/master/ptb/model/ptb_word_lm.py#L203).

## Setup instructions for experiments
Please clone the master branch and follow the instructions to run YellowFin on ResNet for CIFAR10, Bottleneck Resnet on CIRAR100 for image recognition, LSTM on Penn Treebank for language modeling, Char Rnn LSTM on TinyShakespeare and LSTM on Wall Street Journal dataset for constituency parsing. The CIFAR and PTB models we use are slightly adapted from official Tensorflow [ResNet](https://github.com/tensorflow/models/tree/master/resnet) and [LSTM](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb). The Char Rnn LSTM and the Parsing LSTM are adapted from [Char Rnn repo](https://github.com/sherjilozair/char-rnn-tensorflow) and [Parsing LSTM repo](https://github.com/cdg720/emnlp2016) respectively. Thanks to the researchers for developing the models.

Note YellowFin is tested under Tensorflow 1.1 and Python 2.7.

### download data
Please use the data/download.sh script to download CIFAR10/100 and Penn Treebank dataset. It may take a few minutes depending on the network speed. Other datasets are self-included in the repo.
```
cd data
bash download.sh
```

### Run CIFAR10/100 ResNets experiments
The experiments on 110 layer ResNet with CIFAR10 and 164 layer ResNet with CIFAR100 can be launched using
```
cd cifar/scripts
python CIFAR10-release.py (for CIFAR10)
python CIFAR100-release.py (for CIFAR10)
```

### Run Penn Treebank LSTM experiments
The experiments on multiple-layer LSTM on Penn Treebank can be launched using
```
cd ptb/scripts
python PTB-release.py
```

### Run Char Rnn LSTM experiments
The experiments on Char Rnn LSTM with TinyShakespeare dataset can be launched using
```
cd char-rnn-tensorflow
python train_YF.py --log_dir=path_to_log --data_dir=./data/tinyshakespeare/
```

### Run constituency parsing LSTM experiments
The experiments on constituency parsing with the Wall Street Journal (WSJ) dataset can be launched using
```
cd parsing
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=path_to_log --opt_method="YF"
```
Note the WSJ is not public available. Please contact us or the author of [Parsing LSTM repo](https://github.com/cdg720/emnlp2016) for the access of the data. The data can be preprocessed following the instructions in [Parsing LSTM repo](https://github.com/cdg720/emnlp2016). You should be able to run our scripts on the preprocessed data.


## Detailed guidelines
a. YFOptimizer(lr=1.0, mu=0.0) sets initial learnig rate and momentum to 1.0 and 0.0 respectively. This is the uniform setting (i.e. without tuning) for all our PyTorch and Tensorflow experiments. Typically, after a few thousand minibatches, the influence of these initial values diminishes.

b. If you want to clip the gradient, you can also consider using the ```clip_thresh``` argument when initializing the YFOptimizer.

c. If you want to use the typical lr-dropping technique after a ceritain number of epochs, or you want to more finely control the learning rate, please use ```lr_factor``` in the YFOptimizer class. More details can be found [here](https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py#L30). 


### Implementation for other platforms
For PyTorch users, we implemented [YellowFin PyTorch repo](https://github.com/JianGoForIt/YellowFin_Pytorch).

<!---For MXNet users, Github user [StargazerZhu](https://github.com/StargazerZhu) has already implemented a Theano version here: [YellowFin MXNet Repo](https://github.com/StargazerZhu/YellowFin_MXNet).--->

For Theano users, Github user [botev](https://github.com/botev) has already implemented a Theano version here: [YellowFin Theano Repo](https://gist.github.com/botev/f8b32c00eafee222e47393f7f0747666).

We thank the contributors for YellowFin in different deep learning frameworks.
