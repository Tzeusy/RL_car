# All packages installed using pip
# python 3.5.2
# torch 1.1.0
# numpy 1.16.4

import heapq
import random
import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import torch.nn.functional as F
import torch.distributions.categorical as cat

from collections import Counter
from model import Baseline
from itertools import count
from plots import plot_rewards
from enum import Enum
from tqdm import tqdm


class Actions(Enum):
    NOTHING, GAS, BRAKE, LEFT, RIGHT = range(5)


def index_to_action(action_index: int) -> np.ndarray:
    action = np.zeros(3)  # steer, gas, brake

    if action_index == Actions.GAS.value:
        action[1] = 1
    elif action_index == Actions.BRAKE.value:
        action[2] = 0.8
    elif action_index == Actions.LEFT.value:
        action[0] = -1
    elif action_index == Actions.RIGHT.value:
        action[0] = 1
    return action


def action_to_index(action: np.ndarray) -> int:
    if action[0] == -1:
        return Actions.LEFT.value
    if action[0] == 1:
        return Actions.RIGHT.value
    if action[1] == 1:
        return Actions.GAS.value
    if action[2] == 0.8:
        return Actions.BRAKE.value
    return Actions.NOTHING.value


def select_action(state, model, device, steps_done):

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) \
                    * max((1 - steps_done / EPS_DECAY), 0)

    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float, device=device)
            pi, _ = model(state)
            return cat.Categorical(pi).sample().item()
    else:
        return random.randrange(5)


def optimize_model(device, model, optimizer, rewards, actions,
                   states):
    L1, L2 = 0, 0

    # Compute g
    T = len(rewards)
    g = np.zeros(T)
    g[-1] = rewards[-1]
    for i in range(T - 2, -1, -1):
        g[i] = rewards[i] + GAMMA * g[i + 1]
    g = torch.tensor(g, dtype=torch.float, device=device)

    # Compute pi
    states = torch.tensor(states, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.float, device=device)

    pi, v = model(states)
    v = v.squeeze(1)
    actual_log_prob = cat.Categorical(pi)
    actual_log_prob = actual_log_prob.log_prob(actions)

    # Compute L
    for t in range(T):
        L1 += -(GAMMA**t) * (g[t] - v[t].detach()) * actual_log_prob[t]

    L2 = F.smooth_l1_loss(g, v)
    loss = L1 + LOSS2_C * L2

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def generate_episode(env, model, device, steps_done, episode_rewards):
    state = env.reset().transpose((2, 0, 1))
    i_rewards, i_states, i_actions = [], [], []
    total_reward = 0

    for t in count():
        # Select and perform an action
        action_idx = select_action(state, model, device, steps_done)
        action = index_to_action(action_idx)
        new_state, reward, done, _ = env.step(action)
        env.render()
        new_state = new_state.transpose((2, 0, 1))
        steps_done += 1

        # Save reward, action, state
        i_rewards.append(reward)
        i_actions.append(action_idx)
        i_states.append(state)
        total_reward += reward

        # Move state forward
        state = new_state

        # Break if done
        if done or t == 5000:
            print(total_reward)
            episode_rewards.append(total_reward)
            plot_rewards(episode_rewards)
            break

    return i_rewards, i_states, i_actions, steps_done


def create_env():
    env = gym.make('CarRacing-v0').unwrapped
    env.mode = 'fast'
    env.seed(0)
    return env


N_HIDDEN_NODES = 24
N_HIDDEN_LAYERS = 3
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000000
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-6
LOSS2_C = 0.01
SAVE_EPI = 50


def train():
    # Prepare gym
    env = create_env()
    h, w, c = env.observation_space.shape

    # Prepare models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, fn = "./policy_grad", '{}.pth'
    model = Baseline(h, w).to(device)
    model.train()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

    # Train
    steps_done = 0
    num_episodes = 2000
    episode_rewards = []

    for i_episode in tqdm(range(num_episodes)):
        # Complete 1 episode
        print("Episode {}".format(i_episode + 1))
        i_rewards, i_states, i_actions, steps_done = generate_episode(
            env, model, device, steps_done, episode_rewards
        )

        # Update model
        optimize_model(device, model, optimizer, i_rewards,
                       i_actions, i_states)

        # Save model every couple episodes
        if (i_episode + 1) % SAVE_EPI == 0:
            path = os.path.join(model_dir, fn.format(episode_rewards[-1]))
            torch.save(model.state_dict(), path)

    print('Complete')
    np.save('./rewards_policy_grad.npy', episode_rewards)

    env.close()
    plt.ioff()
    plt.show()


def test():
    # Prepare env
    env = create_env()
    h, w, c = env.observation_space.shape

    # Load 5 best models
    device = torch.device("cpu")
    model_dir = "./policy_grad"
    model_fns = {}
    for fn in os.listdir(model_dir):
        if fn.endswith('.pth'):
            score = fn.split("_")[-1][:-4]
            model_fns[fn] = float(score)
    top_5 = heapq.nlargest(3, model_fns, key=model_fns.get)

    models = []
    for fn in top_5:
        path = os.path.join(model_dir, fn)
        model = Baseline(h, w).to(device)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        models.append(model)

    # Watch race car perform
    state = env.reset().transpose((2, 0, 1))
    state = torch.tensor([state], dtype=torch.float, device=device)
    total_reward = 0
    for t in count():
        # Select and perform an action
        votes = []
        for model in models:
            pi, _ = model(state)
            votes.append(pi.argmax().item())
        action_idx = Counter(votes).most_common(1)[0][0]
        action = index_to_action(action_idx)
        state, reward, done, _ = env.step(action)
        env.render()

        # Update
        state = state.transpose((2, 0, 1))
        state = torch.tensor([state], dtype=torch.float, device=device)
        total_reward += reward
        if done:
            break
    print("Total reward: {}".format(total_reward))


if __name__ == '__main__':
    test()
