for seed in 2 1 3
do
  python CIFAR100-release.py --seed=${seed} --log_dir=../results/YF_seed_${seed}_h_max_log_test_slow_start_10_win_h_max_clip_cifar100 --opt_method=YF --h_max_log_smooth 
done
#for seed in 2
#do
#  for lr in 0.001 0.01 0.1 1.0 0.0001
#  do
#    python CIFAR100-release.py --seed=${seed} --log_dir=../results/Adam_lr_${lr}_seed_${seed} --opt_method=adam --lr=${lr}
#  done
#done
