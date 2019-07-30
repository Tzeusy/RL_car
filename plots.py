import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join


def plot_rewards(episode_rewards, save_name, save_dir='plots'):
    assert not save_name.endswith('.jpg'), 'Omit .jpg from save_name'
    os.makedirs(save_dir, exist_ok=True)

    rewards = np.asarray(episode_rewards)
    averages = []
    for i in range(1, len(rewards)):
        if i < 100:
            averages.append(rewards[:i].mean())
        else:
            averages.append(rewards[i-100:i].mean())

    plt.figure()
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')

    plt.plot(rewards, label='Rewards')
    plt.plot(averages, label='Averages')

    plt.savefig(join(save_dir, f'{save_name}.jpg'))
    plt.close('all')
