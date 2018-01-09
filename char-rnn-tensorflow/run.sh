for seed in 2 1 3
do
  python train_YF.py --log_dir=./results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --data_dir=./data/tinyshakespeare/ --opt_method=YF --seed=${seed} --h_max_log_smooth

#  python train.py --log_dir=./results/Adam_0.0005_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.0005 --seed=${seed} --opt_method=Adam
#  python train.py --log_dir=./results/Adam_0.001_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.001 --seed=${seed} --opt_method=Adam
#  python train.py --log_dir=./results/Adam_0.05_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.05 --seed=${seed} --opt_method=Adam
#  python train.py --log_dir=./results/Adam_0.005_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.005 --seed=${seed} --opt_method=Adam
#  python train.py --log_dir=./results/Adam_0.01_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.01 --seed=${seed} --opt_method=Adam

#  python train.py --log_dir=./results/MOM_SGD_0.5_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.5 --seed=${seed} --opt_method=momSGD
#  python train.py --log_dir=./results/MOM_SGD_1.0_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=1.0 --seed=${seed} --opt_method=momSGD
#  python train.py --log_dir=./results/MOM_SGD_0.1_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.1 --seed=${seed} --opt_method=momSGD
#  python train.py --log_dir=./results/MOM_SGD_0.05_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=0.05 --seed=${seed} --opt_method=momSGD
#  python train.py --log_dir=./results/MOM_SGD_5.0_seed_${seed}/ --data_dir=./data/tinyshakespeare/ --learning_rate=5.0 --seed=${seed} --opt_method=momSGD 
done
