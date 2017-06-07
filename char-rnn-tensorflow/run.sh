#python train_YF.py --log_dir=./results/YF/ --data_dir=./data/tinyshakespeare/
#python train.py --log_dir=./results/Adam_0.0005/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.0005 
#python train.py --log_dir=./results/Adam_0.001/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.001
#python train.py --log_dir=./results/Adam_0.002/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.002
#python train.py --log_dir=./results/Adam_0.005/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.005
#python train.py --log_dir=./results/Adam_0.01/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.01
python train.py --log_dir=./results/SGD_1.0/ --data_dir=./data/tinyshakespeare/ --learning_rate=1.0
python train.py --log_dir=./results/SGD_0.5/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.5
python train.py --log_dir=./results/SGD_0.1/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.1
python train.py --log_dir=./results/SGD_0.05/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.05
python train.py --log_dir=./results/SGD_5.0/ --data_dir=./data/tinyshakespeare/ --learning_rate=5.0 
 
