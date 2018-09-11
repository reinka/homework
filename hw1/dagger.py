import load_policy
import pickle
import tensorflow as tf
import tf_util
import numpy as np

from util import train_test_scale, play_env, plt_errorbars
from sklearn.model_selection import train_test_split
from models import BaselineModel

envname = 'Hopper-v2'


def run_expert_on_obs(obs, policy_fn):
    with tf.Session():
        tf_util.initialize()

        actions = []

        for o in obs:
            a = policy_fn(o[None, :])
            actions.append(a)

    return np.array(actions)


initial_data = pickle.load(open('experts/data/1rollouts.pcl', 'rb'))
expert_policy = load_policy.load_policy('experts/Hopper-v1.pkl')

obs = initial_data['observations']
actions = initial_data['actions'][:, 0]

n_iter = 15
n_hidden = 64
n_output = 3
opt = tf.train.AdamOptimizer(learning_rate=0.001)

rewards = {}
num_rolls = 5

bar = tf.keras.utils.Progbar(n_iter)
for i in range(n_iter):
    bar.update(i + 1)
    trX, teX, trY, teY = train_test_split(obs, actions,
                                          test_size=.1)
    # scale data
    trX, teX, trY, teY, scalerX, scalerY = train_test_scale(trX, teX, trY, teY)

    policy = BaselineModel(n_hidden=n_hidden, output_dim=n_output,
                           scalerX=scalerX, scalerY=scalerY)
    policy.compile(loss='mse', optimizer=opt,
                   metrics=['mse'])
    history = policy.fit(trX, trY, batch_size=128, epochs=5,
                         #                          validation_data=[teX, teY],
                         verbose=0)

    results = play_env(envname=envname, model=policy,
                       num_rollouts=num_rolls, render=False)
    policy_obs = results['observations']
    rewards[i] = results['rewards']

    expert_actions = run_expert_on_obs(obs=policy_obs, policy_fn=expert_policy)

    obs = np.append(obs, policy_obs, axis=0)
    actions = np.append(actions, expert_actions.reshape(-1, n_output), axis=0)


plt_errorbars(list(range(1, n_iter + 1)), np.mean(list(rewards.values()), axis=1),
              yerr=np.std(list(rewards.values()), axis=1),
              xlabel='Iteration')