This is the repository of the code for the paper [Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations](https://arxiv.org/abs/2108.01846), accepted at NeurIPS 2021.

# Installation
Create the environment
```
conda create -n safe -y python=3.9
pip install -r requirements.txt
pip install -r lunzi/requirements.txt
```

# How to run

Before we run, if you don't use `wandb`, simply run `wandb offline`.

Suppose we want to run CRABS in the task `Swing`. We also provide intermediate checkpoints.
There are step steps to get CRABS working:

0. Run `export PYTHONPATH=$PYTHONPATH:.`
1. Train an initial barrier certificate. The barrier certificate can be found at `/tmp/crabs/pretrain/wandb/run-xxx/ckpt.pt`.
    > python run/main.py --root_dir /tmp/crabs/pretrain/ -c ./configs/train_h.json5
2. Train a policy iteratively. The policies can be found at `/tmp/crabs/iterative/wandb/run-xxx/ckpt.pt` and the log file can be found at `/tmp/crabs/iterative/wandb/run-xxx/run-xxx.wandb`
    > python run/main.py --root_dir /tmp/crabs/iterative/ -c ./configs/train_policy.json5

To read the log file and generate the plot, please check `run/read_log.py`.
