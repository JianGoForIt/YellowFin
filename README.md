# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tunes momentum as well as learning rate using local quadratic approximation.

The implementation here can be **a drop-in replacement for any optimizer in Tensorflow**. It supports both ```minimize``` and ```apply_gradients``` like any tensorflow optimizer after ```from yellowfin import YFOptimizer```. **We also provide interface to manually set the learning rate schedule at every iteration for finer control (See Detailed Guideline section)**.

For more technical details, please refer to our paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).

For more usage details, please refer to the inline documentation of ```tuner_utils/yellowfin.py```. Example usage can be found here for [CIFAR](https://github.com/JianGoForIt/YellowFin/blob/master/cifar/model/resnet_model.py#L160) and [PTB](https://github.com/JianGoForIt/YellowFin/blob/master/ptb/model/ptb_word_lm.py#L203).

**YellowFin is under active development. Many members of the community have kindly submitted issues and pull requests. We are incorporating fixes and smoothing things out. As a result the repository code is in flux. Please make sure you use the latest version and submit any issues you might have!**

<!---We thank @[mfernezir](https://github.com/mfernezir) for the efforts on standardization of YellowFin in TensorFlow. If you want to use the previous stable version, please check out v1.0 branch.--->


## Updates
**[2017.08.06] Switched to logrithmic smoothing to accelerate adaptation to curvature range trends.**

**[2017.08.06] Added feature to correct estimation bias from sparse gradient.**

**[2017.08.11] Added Multipe GPU training support with better standardized code structure.**

**[2017.08.16] Replace numpy root solver with closed form solution using Vieta's substitution for cubic eqaution. It solves the stability issue of the numpy root solver.**

***[2017.10.29] Major fixe for stability. We added eps to protect fractions in our code, as well as an adaptive clipping feature to properly deal with exploding gradient (manual clipping is still supported as described in the detailed instruction below).***

## Setup instructions for experiments
Please clone the master branch and follow the instructions to run YellowFin on ResNet for CIFAR10, Bottleneck Resnet on CIRAR100 for image recognition, LSTM on Penn Treebank for language modeling, Char Rnn LSTM on TinyShakespeare and LSTM on Wall Street Journal dataset for constituency parsing. The CIFAR and PTB models we use are slightly adapted from official Tensorflow [ResNet](https://github.com/tensorflow/models/tree/master/resnet) and [LSTM](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb). The Char Rnn LSTM and the Parsing LSTM are adapted from [Char Rnn repo](https://github.com/sherjilozair/char-rnn-tensorflow) and [Parsing LSTM repo](https://github.com/cdg720/emnlp2016) respectively. Thanks to the researchers for developing the models.

YellowFin is tested under Tensorflow 1.1 and Python 2.7.

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
python CIFAR10-release.py --log_dir=path_to_log --opt_method=YF (for CIFAR10)
python CIFAR100-release.py --log_dir=path_to_log --opt_method=YF (for CIFAR100)
```

### Run Penn Treebank LSTM experiments
The experiments on multiple-layer LSTM on Penn Treebank can be launched using
```
cd ptb/scripts
python PTB-release.py --opt_method=YF --log_dir=path_to_log
```

### Run Char Rnn LSTM experiments
The experiments on Char Rnn LSTM with TinyShakespeare dataset can be launched using
```
cd char-rnn-tensorflow
python train_YF.py --log_dir=path_to_log --data_dir=./data/tinyshakespeare/ --opt_method=YF
```

### Run constituency parsing LSTM experiments
The experiments on constituency parsing with the Wall Street Journal (WSJ) dataset can be launched using
```
cd parsing
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=path_to_log --opt_method="YF"
```
Note the WSJ is not public available. Please contact us or the author of [Parsing LSTM repo](https://github.com/cdg720/emnlp2016) for the access of the data. The data can be preprocessed following the instructions in [Parsing LSTM repo](https://github.com/cdg720/emnlp2016). You should be able to run our scripts on the preprocessed data.


## Detailed guidelines
* **Basic use**: YFOptimizer() uses the uniform setting (i.e. without tuning) for all the PyTorch and Tensorflow experiments in our paper. 

* **Interface for manual finer control**: If you want to more finely control the learning rate, please use ```lr_factor``` in the YFOptimizer class. E.g. if you want to use a manually set constant learning rate, you can assign ```desired_lr / self._lr_var``` to ```self.lr_factor``` before applying the gradient at each iteration. If you want to use the typical lr-dropping technique after a ceritain number of epochs, please refer to the example [here](https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L139). **(The argument ```learning_rate``` and ```momentum``` are dummy, only for backward compatibility)**

* **Gradient clipping**: The default setting uses adaptive gradient clipping to prevent gradient explosion, thresholding norm of gradient to the square root of our estimated maximal curvature. We recommend first fully turning off gradient clipping, and only turning it on when necessary. 

  * If you want to set the clipping threshold manually, please first use ```use_adapt_grad_clip=False``` when initializing the YFOptimmizer to turn off the adaptive clipping. You may use the ```clip_thresh=thresh_norm_of_gradient``` argument when initializing the YFOptimizer to threshold the norm of gradient, or you can do the gradient clipping outside of YFOptimizer. 
  * if you want to fully turn off gradient clipping inside YFOptimmizer, please set ```use_adapt_grad_clip=False``` when initializing YFOptimizer.

* **Normalization**: When using log probability style losses, please make sure the loss is properly normalized. In some RNN/LSTM cases, the cross_entropy need to be averaged by the number of samples in a minibatch. Sometimes, it also needs to be averaged over the number of classes and the sequence length of each sample in some Tensorflow loss functions. E.g. the cross_etropy loss [here](https://github.com/JianGoForIt/YellowFin/blob/master/ptb/model/ptb_word_lm.py#L168) need to be normalized by the length of sequence and minibatch size.

<!--- * **Sparsity**: Gradient norm, curvature estimations etc., when calculated with sparse gradient, are biased to larger values than the counterpart from the dense gradient on the full dataset. The bias can be illustrated using the following example: the norm of vectors (1.0, 0.0), (0.0, 1.0) and the norm of their average (0.5, 0.5). The norm of the latter is sqrt(sparsity (i.e. 0.5 here) ) * the norm of the former. The sparsity debias feature is useful when the model is very sparse, e.g. LSTM with word embedding. For non-sparse models, e.g. CNN, turning this feature off could slightly speedup. --->

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

## Acknowledgement
We thank Jack Hessel and Mladen Fernežir for contributing to the codebase. 

### Implementation for other platforms
For PyTorch users, we implemented [YellowFin PyTorch repo](https://github.com/JianGoForIt/YellowFin_Pytorch).

<!---For MXNet users, Github user [StargazerZhu](https://github.com/StargazerZhu) has already implemented a Theano version here: [YellowFin MXNet Repo](https://github.com/StargazerZhu/YellowFin_MXNet).--->

<!---For Theano users, Github user [botev](https://github.com/botev) has already implemented a Theano version here: [YellowFin Theano Repo](https://gist.github.com/botev/f8b32c00eafee222e47393f7f0747666).--->

We thank the contributors for YellowFin in different deep learning frameworks.
