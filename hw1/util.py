import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def train_test_scale(trX, teX, trY, teY):
    scalerX, scalerY = StandardScaler().fit(trX), StandardScaler().fit(trY)

    trX = scalerX.transform(trX)
    teX = scalerX.transform(teX)

    trY = scalerY.transform(trY)
    teY = scalerY.transform(teY)

    return trX, teX, trY, teY, scalerX, scalerY


def policy_fn(model, obs):
    if model.scalerX and model.scalerY:
        action = model.predict(model.scalerX.transform(obs))
        return model.scalerY.inverse_transform(action)

    return model.predict(obs)


def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def play_env(envname, model, policy_fn=policy_fn, num_rollouts=5,
             render=False):
    env = gym.make(envname)
    max_steps = env.spec.timestep_limit
    rewards = []
    observations = []

    for i_episode in range(num_rollouts):
        done = False
        obs = env.reset()
        total_reward = 0
        steps = 0
        while not done:
            observations.append(obs)
            if render:
                env.render()
            action = policy_fn(model, obs[np.newaxis,])

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if steps >= max_steps:
                break
        rewards.append(total_reward)

    return {'rewards': np.array(rewards),
            'observations': np.array(observations)}
