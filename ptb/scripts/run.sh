for seed in 2 1 3
do
  python PTB-release.py --seed=${seed} --opt_method=YF --log_dir=../results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip --h_max_log_smooth
#  python PTB-release.py --seed=${seed} --opt_method=YF --log_dir=../results/YF_seed_${seed}_h_max_linear_h_min_log_lr_t_mu_t 
done

# for random seed sweeping
#for seed in 2 1
#do
#  python PTB-release.py --seed=${seed} --opt_method=YF --log_dir=../results/YF_seed_${seed}  
#done
#for seed in 2 1 3
#do
#  python PTB-release.py --seed=${seed} --opt_method=adam --log_dir=../results/Adam_lr_0.001_seed_${seed} --lr=0.001
#  python PTB-release.py --seed=${seed} --opt_method=adam --log_dir=../results/Adam_lr_0.0001_seed_${seed} --lr=0.0001
#  python PTB-release.py --seed=${seed} --opt_method=adam --log_dir=../results/Adam_lr_1.0_seed_${seed} --lr=1.0
#  python PTB-release.py --seed=${seed} --opt_method=adam --log_dir=../results/Adam_lr_0.1_seed_${seed} --lr=0.1
#  python PTB-release.py --seed=${seed} --opt_method=adam --log_dir=../results/Adam_lr_0.01_seed_${seed} --lr=0.01
#
#  python PTB-release.py --seed=${seed} --opt_method=sgd --log_dir=../results/SGD_lr_0.001_seed_${seed} --lr=0.001
#  python PTB-release.py --seed=${seed} --opt_method=sgd --log_dir=../results/SGD_lr_0.01_seed_${seed} --lr=0.01
#  python PTB-release.py --seed=${seed} --opt_method=sgd --log_dir=../results/SGD_lr_0.1_seed_${seed} --lr=0.1
#  python PTB-release.py --seed=${seed} --opt_method=sgd --log_dir=../results/SGD_lr_1.0_seed_${seed} --lr=1.0
#  python PTB-release.py --seed=${seed} --opt_method=sgd --log_dir=../results/SGD_lr_10.0_seed_${seed} --lr=10.0
#
#done
