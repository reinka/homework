import argparse
import gym
import logz
import numpy as np
import os
import tensorflow as tf
import time

import nn
from sac import SAC
import utils

from multiprocessing import Process

def train_SAC(env_name, exp_name, n_iter, ep_len, seed, logdir, alpha,
              prefill_steps, discount, batch_size, learning_rate, tau, two_qf):
    alpha = {
        'Ant-v2': 0.1,
        'HalfCheetah-v2': 0.2,
        'Hopper-v2': 0.2,
        'Humanoid-v2': 0.05,
        'Walker2d-v2': 0.2,
    }.get(env_name, alpha)

    algorithm_params = {
        'alpha': alpha,
        'batch_size': batch_size,
        'discount': discount,
        'learning_rate': learning_rate,
        'reparameterize': True,
        'tau': tau,
        'epoch_length': ep_len,
        'n_epochs': n_iter,
        'two_qf': two_qf,
    }
    sampler_params = {
        'max_episode_length': 1000,
        'prefill_steps': prefill_steps,
    }
    replay_pool_params = {
        'max_size': 1e6,
    }

    value_function_params = {
        'hidden_layer_sizes': (64, 64),
    }

    q_function_params = {
        'hidden_layer_sizes': (64, 64),
    }

    policy_params = {
        'hidden_layer_sizes': (64, 64),
    }

    logz.configure_output_dir(logdir)
    params = {
        'exp_name': exp_name,
        'env_name': env_name,
        'algorithm_params': algorithm_params,
        'sampler_params': sampler_params,
        'replay_pool_params': replay_pool_params,
        'value_function_params': value_function_params,
        'q_function_params': q_function_params,
        'policy_params': policy_params
    }
    logz.save_params(params)

    env = gym.envs.make(env_name)
    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    sampler = utils.SimpleSampler(**sampler_params)
    replay_pool = utils.SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        **replay_pool_params)

    q_function = nn.QFunction(name='q_function', **q_function_params)
    if algorithm_params.get('two_qf', False):
        q_function2 = nn.QFunction(name='q_function2', **q_function_params)
    else:
        q_function2 = None
    value_function = nn.ValueFunction(
        name='value_function', **value_function_params)
    target_value_function = nn.ValueFunction(
        name='target_value_function', **value_function_params)
    policy = nn.GaussianPolicy(
        action_dim=env.action_space.shape[0],
        reparameterize=algorithm_params['reparameterize'],
        **policy_params)

    sampler.initialize(env, policy, replay_pool)

    algorithm = SAC(**algorithm_params)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True  # may need if using GPU
    with tf.Session(config=tf_config):
        algorithm.build(
            env=env,
            policy=policy,
            q_function=q_function,
            q_function2=q_function2,
            value_function=value_function,
            target_value_function=target_value_function)

        for epoch in algorithm.train(sampler, n_epochs=algorithm_params.get('n_epochs', 1000)):
            logz.log_tabular('Iteration', epoch)
            for k, v in algorithm.get_statistics().items():
                logz.log_tabular(k, v)
            for k, v in replay_pool.get_statistics().items():
                logz.log_tabular(k, v)
            for k, v in sampler.get_statistics().items():
                logz.log_tabular(k, v)
            logz.dump_tabular()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_iter', '-n', type=int, default=500)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    parser.add_argument('--alpha', '-a', type=float, default=0.2)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--discount', '-d', type=float, default=0.99)
    parser.add_argument('--tau', '-t', type=float, default=0.005)
    parser.add_argument('--batch_size', '-bs', type=int, default=256)
    parser.add_argument('--prefill_steps', '-ps', type=int, default=1000)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--one_qf', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'sac_' + args.env_name + '_' + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_SAC(
                env_name=args.env_name,
                exp_name=args.exp_name,
                n_iter=args.n_iter,
                ep_len=args.ep_len,
                seed=seed,
                logdir=os.path.join(logdir, '%d' % seed),
                alpha=args.alpha,
                discount=args.discount,
                prefill_steps=args.prefill_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                tau=args.tau,
                two_qf=not args.one_qf,
            )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
