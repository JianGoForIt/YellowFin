# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tunes momentum as well as learning rate using local quadratic approximation.

The implementation here can be **a drop-in replacement for any optimizer in Tensorflow**. It supports both ```minimize``` and ```apply_gradients``` like any tensorflow optimizer after ```from yellowfin import YFOptimizer```. We also provide interface to manually control the learning rate for fine-tuning.

For more technical details, please refer to our paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).

For more usage details, please refer to the inline documentation of ```tuner_utils/yellowfin.py```. Example usage can be found here for [CIFAR](https://github.com/JianGoForIt/YellowFin/blob/master/cifar/model/resnet_model.py#L160) and [PTB](https://github.com/JianGoForIt/YellowFin/blob/master/ptb/model/ptb_word_lm.py#L203).

**YellowFin is under active development. Many members of the community have kindly submitted issues and pull requests. We are incorporating fixes and smoothing things out. As a result the repository code is in flux. Please make sure you use the latest version and submit any issues you might have!**

We thank @[mfernezir](https://github.com/mfernezir) for the efforts on standardization of YellowFin in TensorFlow. If you want to use the previous stable version, please check out v1.0 branch.


## Updates
**[2017.08.06] Switched to logrithmic smoothing to accelerate adaptation to curvature range trends.**

**[2017.08.06] Added feature to correct estimation bias from sparse gradient.**

**[2017.08.11] Added Multipe GPU training support with better standardized code structure.**

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
* **Basic use**: YFOptimizer(lr=1.0, mu=0.0) sets initial learnig rate and momentum to 1.0 and 0.0 respectively. This is the uniform setting (i.e. without tuning) for all our PyTorch and Tensorflow experiments. Typically, after a few thousand minibatches, the influence of these initial values diminishes.

  * If the loss explodes after a very small number of iterations, you may want to lower the init lr to prevent the explosion at the beginining. 
  
  * We also have users reporting to use regularizer to avoid explosions.

* **Interface for manual finer control**: If you want to more finely control the learning rate, please use ```lr_factor``` in the YFOptimizer class. E.g. if you want to use a manually set constant learning rate, you can assign ```desired_lr / self._lr_var``` to ```self.lr_factor``` before applying the gradient at each iteration. If you want to use the typical lr-dropping technique after a ceritain number of epochs, please refer to the example [here](https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L139). 

* **Gradient clipping**: The default setting will not do gradient clipping to prevent gradient explosion. If you want to clip the gradient, please consider using the ```clip_thresh``` argument when initializing the YFOptimizer to threshold the norm of gradient. We recommend first turning off gradient clipping, which is the default setting, and only turning it on when necessary. 

* **Normalization**: When using log probability style losses, please make sure the loss is properly normalized. In some RNN/LSTM cases, the cross_entropy need to be averaged by the number of samples in a minibatch. Sometimes, it also needs to be averaged over the number of classes and the sequence length of each sample in some Tensorflow loss functions. E.g. the cross_etropy loss [here](https://github.com/JianGoForIt/YellowFin/blob/master/ptb/model/ptb_word_lm.py#L168) need to be normalized by the length of sequence and minibatch size.

* **Sparsity**: Gradient norm, curvature estimations etc., when calculated with sparse gradient, are biased to larger values than the counterpart from the dense gradient on the full dataset. The bias can be illustrated using the following example: the norm of vectors (1.0, 0.0), (0.0, 1.0) and the norm of their average (0.5, 0.5). The norm of the latter is sqrt(sparsity (i.e. 0.5 here) ) * the norm of the former. The sparsity debias feature is useful when the model is very sparse, e.g. LSTM with word embedding. For non-sparse models, e.g. CNN, turning this feature off could slightly speedup.

## Citation
If you use YellowFin in your paper, please cite the paper:
```
@article{zhang2017yellowfin,
  title={YellowFin and the Art of Momentum Tuning},
  author={Zhang, Jian and Mitliagkas, Ioannis and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:1706.03471},
  year={2017}
}
```

### Implementation for other platforms
For PyTorch users, we implemented [YellowFin PyTorch repo](https://github.com/JianGoForIt/YellowFin_Pytorch).

<!---For MXNet users, Github user [StargazerZhu](https://github.com/StargazerZhu) has already implemented a Theano version here: [YellowFin MXNet Repo](https://github.com/StargazerZhu/YellowFin_MXNet).--->

For Theano users, Github user [botev](https://github.com/botev) has already implemented a Theano version here: [YellowFin Theano Repo](https://gist.github.com/botev/f8b32c00eafee222e47393f7f0747666).

We thank the contributors for YellowFin in different deep learning frameworks.
