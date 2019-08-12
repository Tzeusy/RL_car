import matplotlib.pyplot as plt
import torch


def plot_rewards(episode_rewards, title):
    plt.figure(2)
    plt.clf()
    rewards = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
