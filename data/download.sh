# prepare PTB data
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar zxvf simple-examples.tgz
mkdir ptb
cp -r simple-examples/data ptb/
# prepare CIFAR data
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
tar -zxvf cifar-100-binary.tar.gz
mv cifar-10-batches-bin cifar10 
