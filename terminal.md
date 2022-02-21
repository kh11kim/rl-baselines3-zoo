# env_kwargs
render=[True, False], dim=[2], reward_type=["task", "joint"], random_init=[True,False]

# plot
python scripts/plot_train.py -a tqc -e RxbotReach-v0  -f rl-trained-agents/ -w 400 -x steps -y success
python scripts/all_plots.py -a tqc --env RxbotReach-v0 -f rl-trained-agents/

# train
python train.py --algo sac --env RxbotReach-v0 --env-kwargs render:False 
python train.py --algo sac --env RxbotReach-v0 -n 20000 --env-kwargs render:False reward_type:'task' task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:np.pi*6/4
python train.py --algo sac --env RxbotReach-v0 -n 30000 --env-kwargs dim:4 render:False reward_type:'task' task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:np.pi*6/4

python train.py --algo sac --env PandaReach-v0 -n 1000000 --env-kwargs render:False reward_type:"'task'" task_ll:[-1,-1,0] task_ul:[1,1,1]
python train.py --algo sac --env PandaReach-v0 -i logs/sac/PandaReach-v0_1/PandaReach-v0.zip -n 1000000 --env-kwargs render:False reward_type:"'task'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo sac --env PandaReach-v0 -i logs/sac/PandaReach-v0_3/PandaReach-v0.zip -n 1000000 --env-kwargs render:False reward_type:"'task'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo sac --env PandaReach-v0 -n 1000000 --env-kwargs render:False reward_type:"'taskjoint'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo sac --env PandaReach-v0 -n 1000000 --env-kwargs render:False reward_type:"'taskcol'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer

python train.py --algo sac --env PandaReach-v0 -n 1000000 --env-kwargs render:False reward_type:"'taskcoljoint'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo tqc --env PandaReach-v0 -n 1000000 --env-kwargs render:False reward_type:"'taskcol'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo tqc --env PandaReach-v0 -n 2000000 --env-kwargs render:False reward_type:"'taskcoljoint'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo tqc --env PandaReach-v0 -n 2000000 --env-kwargs render:False reward_type:"'taskcolaction'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
python train.py --algo tqc --env PandaReachPosOrn-v0 -n 1000000 --env-kwargs render:False reward_type:"'posorncol'" task_ll:[-1,-1,0] task_ul:[1,1,1] --save-replay-buffer
# hyperparam tuning
python train.py --algo sac --env PandaReach-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median

# enjoy
python enjoy.py --algo sac --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 11 --env-kwargs render:True task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:np.pi*6/4
python enjoy.py --algo sac --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 12 --env-kwargs render:True 
python enjoy.py --algo sac --env PandaReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 1 --env-kwargs render:True task_ll:[-1,-1,0] task_ul:[1,1,1] 
python enjoy.py --algo tqc --env PandaReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 3 --env-kwargs render:True task_ll:[-1,-1,0] task_ul:[1,1,1] 
python enjoy.py --algo tqc --env PandaReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 4 --env-kwargs render:True task_ll:[-1,-1,0] task_ul:[1,1,1] 

python enjoy.py --algo sac --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --exp-id 15 --env-kwargs dim:4 render:True task_ll:[-1,-1,0] task_ul:[1,1,1] joint_range:np.pi*6/4

python enjoy.py --algo tqc --env RxbotReach-v0 --folder rl-trained-agents/ -n 5000 --env-kwargs render:True --load-best