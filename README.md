# Offline Deep Reinforcement Learning for Dynamic Pricing of Consumer Credit

Goal: Implement the proposed approach based in CQL for the credit score problem and replicate the results.

Paper: https://arxiv.org/abs/2203.03003

[Presentation](https://docs.google.com/presentation/d/1zI5DJ_KJri7xAKyH8aohcyYgQ6dHf-gTxk6dNvSkWmA/edit?usp=sharing)

Team:

Alexandra Ivanova\
Andrey Galichin

### Prerequisites

Our code is GPU-based, so we require you to have a gpu compatible with **CUDA 11.3** or less cudatoolkit version.

To construct the environment, you can use any package manager you want. Required packages can be installed via *pip* as follows:

```
pip install -r requirements.txt
```

### Training

Training the model is done as simple as:

```
python train.py
```

You can change the default parameters passed to `train.py` if you want. For more information about available parameters run `python train.py -h`.
You can access training results in default `d3rlpy_logs/` directory or the one specified by `logdir` flag.
