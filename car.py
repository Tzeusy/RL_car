import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time
import os
import sys

from model import DQN, ReplayMemory, Transition

env = gym.make('CarRacing-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

def get_screen():
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

_ = env.reset()

BATCH_SIZE = 64
GAMMA = 0.95
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 30
MAX_EPISODE_LENGTH = 2000
LEARNING_RATE = 5e-3
REPLAY_MEM = 60000

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = 5 #env.action_space.shape[0]

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
print(policy_net)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999999)
memory = ReplayMemory(REPLAY_MEM)
fake_memory = ReplayMemory(REPLAY_MEM)

steps_done = 0

episode_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    selected_action = random.randint(0, n_actions-1)
    # Nothing, Forward, Brake, Left, Right
    selected_action = random.choices(range(5), [0.05, 0.43, 0.004, 0.24, 0.24])[0]

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = policy_net(state).max(1)[1].item()

    return selected_action


def plot_rewards():
    plt.figure()
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(f"./car_{LEARNING_RATE}_{REPLAY_MEM}.png")
    plt.close('all')

def optimize_model():
    if len(memory) < BATCH_SIZE or len(fake_memory) < BATCH_SIZE:
        return
    transitions = memory.sample(int(BATCH_SIZE/2)) + fake_memory.sample(int(BATCH_SIZE/2))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print(target_net(non_final_next_states).size()) # [128, 4]
    max_idx = policy_net(non_final_next_states).max(1)[1]

    # DDQN
    next_state_values[non_final_mask] = torch.gather(target_net(non_final_next_states), dim=1, index=max_idx[:, None]).squeeze()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    scheduler.step()


def train():
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    save_every = 100 # Save every 100 episodes
    global steps_done
    num_episodes = 3000
    # plt.figure()
    action_direction = ['Nothing', 'Gas', 'Brake', 'Left', 'Right']
    
    from pyglet.window import key
    fake_action = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        print(f"Key {k} pressed")
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  fake_action[0] = -1.0
        if k==key.RIGHT: fake_action[0] = +1.0
        if k==key.UP:    fake_action[1] = +1.0
        if k==key.DOWN:  fake_action[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and fake_action[0]==-1.0: fake_action[0] = 0
        if k==key.RIGHT and fake_action[0]==+1.0: fake_action[0] = 0
        if k==key.UP:    fake_action[1] = 0
        if k==key.DOWN:  fake_action[2] = 0
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    for i_episode in range(num_episodes):
        steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        # Initialize the environment and state
        _ = env.reset()
        start = time.time()
        
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        total_reward = 0
        consecutive_noreward = 0
        action_barchart = np.zeros(n_actions)
        
        for t in count():
            fake_action_val = 0
            if fake_action[0] == -1:
                fake_action_val = 3
            if fake_action[0] == 1:
                fake_action_val = 4
            if fake_action[1] == 1:
                fake_action_val = 1
            if fake_action[2] == 0.8:
                fake_action_val = 2
            if fake_action_val != 0:
                print("Inputting fake action for imitation")
            fake_action_tensor = torch.tensor([[fake_action_val]], device=device, dtype=torch.long)
            # Select and perform an action
            selected_action = select_action(state)
            # print(selected_action)
            action_barchart[selected_action] += 1
            # if t%10 == 0:
            #     plt.clf()
            #     plt.pause(0.01)
            #     plt.bar(action_direction, action_barchart)
            #     plt.show()
                # print(action_histogram)
            real_action = np.zeros((3)) # Steer, gas, brake
            real_action[1] = 0.1
            if selected_action == 0:
                # Do nothing
                pass
            if selected_action == 1:
                # Gas only
                real_action[1] = 1
            elif selected_action == 2:
                # Brake only
                real_action[2] = 0.8
            elif selected_action == 3:
                # Left
                real_action[0] = -1
            elif selected_action == 4:
                # Right
                real_action[0] = 1
            # action_tensor = torch.tensor(real_action, device=device, dtype=torch.long)
            real_action_tensor = torch.tensor([[selected_action]], device=device, dtype=torch.long)
            if fake_action_val == 0:
                _, reward, done, _ = env.step(real_action)
            else:
                _, reward, done, _ = env.step(fake_action)
            if(reward<0):
                consecutive_noreward += 1
            else:
                consecutive_noreward = 0

            if(consecutive_noreward > 50):
                reward = -200
                done = True

            if (selected_action != fake_action_val) and fake_action_val != 0:
                reward -= 5
            elif (selected_action == fake_action_val) and fake_action_val != 0:
                reward += 5

            total_reward += reward

            reward = torch.tensor([reward], dtype=torch.float, device=device)
            
            if(i_episode%1)==0:
                env.render()

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            # memory.push(state, fake_action_tensor, next_state, reward)
            if fake_action_val != 0:
                fake_memory.push(state, fake_action_tensor, next_state, torch.tensor([1], dtype=torch.float, device=device))
            memory.push(state, real_action_tensor, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()

            if (done or t > MAX_EPISODE_LENGTH):
                state = None
                print(f"Episode {i_episode} with {t} length took {time.time()-start}s and scored {total_reward}")
                episode_rewards.append(total_reward)
                if i_episode % 20 == 0:
                    plot_rewards()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        current_reward = np.mean(episode_rewards[-100:-1])

        if i_episode % save_every == 0 and i_episode > 0:
            torch.save(target_net.state_dict(), f"{snapshot_dir}/target_episode{i_episode}_reward_{current_reward:.3f}.pth")

    print('Complete')
    # env.render()
    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    train()

