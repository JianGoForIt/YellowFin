#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_0.25 --opt_method="SGD"
#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/YF --opt_method="YF"

#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/Adam_0.001 --opt_method="Adam" --learning_rate=0.001
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/Adam_0.005 --opt_method="Adam" --learning_rate=0.005
sleep 1m
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/Adam_0.01 --opt_method="Adam" --learning_rate=0.01
sleep 1m 
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/Adam_0.0005 --opt_method="Adam" --learning_rate=0.0005
sleep 1m
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/Adam_0.0001 --opt_method="Adam" --learning_rate=0.0001

#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_1.0 --opt_method="SGD" --learning_rate=1.0
#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_5.0 --opt_method="SGD" --learning_rate=5.0
#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_0.5 --opt_method="SGD" --learning_rate=0.5
#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_0.1 --opt_method="SGD" --learning_rate=0.1
#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_0.05 --opt_method="SGD" --learning_rate=0.05

