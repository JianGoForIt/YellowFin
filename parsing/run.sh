#mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./test --opt_method="SGD"
mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model --log_dir=./test --opt_method="YF"
