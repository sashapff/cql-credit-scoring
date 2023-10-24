import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import d3rlpy
from sklearn.model_selection import train_test_split

import time

from src.utils import (fill_unknown_previous_rate, split_on_observations_nd_actions, 
                    get_acceptance_model)
from src.reward import reward
from src.environment import Environment


def _main(cfg):
    data_dir = Path(cfg.data_dir)

    train = pd.read_csv(data_dir / 'train_data.csv')
    test = pd.read_csv(data_dir / 'test_data.csv')

    prev_rate_mean = fill_unknown_previous_rate(train)
    fill_unknown_previous_rate(test, prev_rate_mean)

    print('Training acceptence model...')
    accept_model = get_acceptance_model(train)
    print('Done.')

    train_data, eval_data = train_test_split(
        train[:cfg.n_samples], test_size=0.1, shuffle=True, random_state=42)
    
    train_observations, train_actions = split_on_observations_nd_actions(train_data)
    eval_observations, eval_actions = split_on_observations_nd_actions(eval_data)

    print('Calculating rewards...')
    train_rewards = reward(train_actions, train_observations, accept_model, risk_free=0.04, loss_ratio=0.5)
    eval_rewards = reward(eval_actions, eval_observations, accept_model, risk_free=0.04, loss_ratio=0.5)
    print('Done.')

    # Initialize environment
    environment = Environment(train_data, accept_model)

    # Initialize train dataset
    train_dataset = d3rlpy.dataset.MDPDataset(
        observations=train_observations,
        actions=train_actions,
        rewards=train_rewards,
        terminals=np.random.randint(2, size=train_actions.shape[0]),
    )

    # Initialize evaluation dataset
    eval_dataset =d3rlpy.dataset.MDPDataset(
        observations=eval_observations,
        actions=eval_actions,
        rewards=eval_rewards,
        terminals=np.zeros_like(eval_actions)
    )

    # Initialize CQL algorithm
    cql = d3rlpy.algos.CQL(use_gpu=True, n_steps=5, batch_size=cfg.batch_size, 
                           gamma=cfg.discount_factor, n_critics=2, 
                           alpha_threshold=10, conservative_weight=5)

    print(f'Action size = ', train_dataset.get_action_size())
    time.sleep(0.1)

    # Train
    print('Starting training...')
    cql.fit(train_dataset,
            eval_episodes=train_dataset,
            n_epochs=cfg.n_epochs,
            scorers={'environment': d3rlpy.metrics.evaluate_on_environment(environment),
                    'td_error': d3rlpy.metrics.td_error_scorer,},
            logdir=cfg.logdir)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', default='data',
                        help='Directory where the dataset is located')
    parser.add_argument('--logdir', default='d3rlpy_logs',
                        help='Specify directory where cql model and logs will be stored')
    parser.add_argument('--n_samples', default=1000, 
                        help='Number of samples to use in cql training')
    parser.add_argument('--discount_factor', default=0.999, 
                        help='Discount factor for cql')
    parser.add_argument('--n_epochs', default=1, help='Number of epochs for cql')
    parser.add_argument('--batch_size', default=128, 
                        help='batch size for cql algorithm')

    args = parser.parse_args()
    _main(args)
