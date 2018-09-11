import argparse
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split

from models import BaselineModel
from util import play_env, plot_loss, policy_fn, train_test_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("hidden_dim", type=int,
                        help='Number of hidden units.')
    parser.add_argument("output_dim", type=int,
                        help='Number of output units.')
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of policy roll outs')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size.')
    parser.add_argument('--plot_loss', action='store_true')

    args = parser.parse_args()
    print('Loading expert data.')
    data = pickle.load(open(args.expert_data_file, 'rb'))
    obs = pd.DataFrame(data['observations'])
    actions = pd.DataFrame(data['actions'][:, 0])

    print('Data info of observations')
    obs.info()
    print()
    print('Data info of actions')
    actions.info()

    trX, teX, trY, teY = train_test_split(obs.values, actions.values,
                                          test_size=.1)

    # scale data
    trX, teX, trY, teY, scalerX, scalerY = train_test_scale(trX, teX, trY, teY)

    baseline_model = BaselineModel(args.hidden_dim, args.output_dim,
                                   scalerX=scalerX, scalerY=scalerY)
    baseline_model.compile(loss='mse',
                           optimizer=tf.train.AdamOptimizer(
                               learning_rate=0.001),
                           metrics=['mse'])

    history = baseline_model.fit(trX, trY, batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 validation_data=[teX, teY], verbose=1)

    try:
        baseline_model.summary()
    except ValueError as e:
        print(e)
    if args.plot_loss:
        plot_loss(history)

    results = play_env(args.envname, baseline_model, policy_fn,
                       args.num_rollouts, args.render)

    print('returns', results['rewards'])
    print('mean return', np.mean(results['rewards']))
    print('std of return', np.std(results['rewards']))


if __name__ == '__main__':
    main()
