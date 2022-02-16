# env_kwargs
render=[True, False], dim=[2], reward_type=["task", "joint"], random_init=[True,False]

# plot
python scripts/plot_train.py -a tqc -e RxbotReach-v0  -f rl-trained-agents/ -w 400 -x steps -y success
python scripts/all_plots.py -a tqc --env RxbotReach-v0 -f rl-trained-agents/

# train
python train.py --algo sac --env RxbotReach-v0 --env-kwargs render:False 
python train.py --algo sac --env RxbotReach-v0 --env-kwargs render:False task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:6.28


python enjoy.py --algo sac --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 3 --env-kwargs render:True task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:np.pi*2

python enjoy.py --algo tqc --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --env-kwargs render:True --load-best