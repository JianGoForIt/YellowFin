#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/SGD_0.25 --opt_method="SGD"
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./results/YF --opt_method="YF"
