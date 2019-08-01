import gym
import math
import random
import numpy as np
from itertools import count
from PIL import Image
from pyglet.window import key
from enum import Enum

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time
import os

from model import DQN, ReplayMemory, Transition
from plots import plot_rewards


class Actions(Enum):
    NOTHING, GAS, BRAKE, LEFT, RIGHT = range(5)


env = gym.make('CarRacing-v0').unwrapped
env.reset()


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


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 30
NUM_EPISODES = 3_000
MAX_EPISODE_LENGTH = 2_000
LEARNING_RATE = 5e-3
REPLAY_MEM = 90_000

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = 5  # env.action_space.shape[0]

snapshot_dir = "snapshots"
model_dir = "semi_successful_models"
model_file_name = "manual30_ep40.pth"

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load(f"{model_dir}/{model_file_name}"))
policy_net.eval()

target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999999)
memory = ReplayMemory(REPLAY_MEM)
fake_memory = ReplayMemory(REPLAY_MEM)

steps_done = 0


def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    # selected_action = random.randint(0, n_actions-1)
    # Nothing, Forward, Brake, Left, Right
    selected_action = random.choices(range(5), [0.05, 0.45, 0.002, 0.24, 0.24])[0]

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = policy_net(state).max(1)[1].item()

    return selected_action


def optimize_model():
    if len(memory) < BATCH_SIZE or len(fake_memory) < BATCH_SIZE:
        return
    if len(fake_memory) >= BATCH_SIZE:
        transitions = memory.sample(BATCH_SIZE // 2) + fake_memory.sample(BATCH_SIZE // 2)
    else:
        transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                  dtype=torch.uint8, device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

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
    os.makedirs(snapshot_dir, exist_ok=True)
    save_every = 20  # Save every 100 episodes

    fake_action = np.zeros(3)
    fake_action_listener(env, fake_action)

    episode_rewards = []
    for i_episode in range(NUM_EPISODES):
        global steps_done
        steps_done += 1
        # Initialize the environment and state
        env.reset()
        start = time.time()
        
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        total_reward = 0
        consecutive_noreward = 0

        for t in count():
            real_action_idx = select_action(state)
            real_action = index_to_action(real_action_idx)

            fake_action_idx = action_to_index(fake_action)
            fake_action_available = fake_action_idx != Actions.NOTHING.value
            if fake_action_available:
                print('Inputting fake action for imitation')

            if fake_action_available:
                _, reward, done, _ = env.step(fake_action)
            else:
                _, reward, done, _ = env.step(real_action)

            if(reward<0):
                consecutive_noreward += 1
            else:
                consecutive_noreward = 0
            if(consecutive_noreward > 50):
                if total_reward < 750:
                    reward -= 100
                done = True

            if (real_action_idx != fake_action_idx) and fake_action_available:
                reward -= 5
            elif (real_action_idx == fake_action_idx) and fake_action_available:
                reward += 5
            total_reward += reward

            if(i_episode%20 == 0) or i_episode < 100:
                env.render()

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            real_action_tensor = torch.tensor([[real_action_idx]], dtype=torch.long, device=device)
            fake_action_tensor = torch.tensor([[fake_action_idx]], dtype=torch.long, device=device)
            if fake_action_available:
                fake_reward = torch.tensor([5], dtype=torch.float, device=device)
                fake_memory.push(state, fake_action_tensor, next_state, fake_reward)
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
                    plot_rewards(episode_rewards, save_name=f'car_{LEARNING_RATE}_{REPLAY_MEM}')
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        current_reward = np.mean(episode_rewards[-100:-1])
        if i_episode % save_every == 0 and i_episode > 0:
            filename = f'{snapshot_dir}/target_episode{i_episode}_reward_{current_reward:.3f}.pth'
            torch.save(target_net.state_dict(), filename)

    print('Complete')
    env.close()


def fake_action_listener(env, fake_action):
    def key_press(k, mod):
        if k == key.LEFT:
            fake_action[0] = -1.0
        elif k == key.RIGHT:
            fake_action[0] = 1.0
        elif k == key.UP:
            fake_action[1] = 1.0
        elif k == key.DOWN:
            fake_action[2] = 0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and fake_action[0] == -1.0:
            fake_action[0] = 0
        elif key.RIGHT and fake_action[0] == 1.0:
            fake_action[0] = 0
        elif k == key.UP:
            fake_action[1] = 0.0
        elif k == key.DOWN:
            fake_action[2] = 0.0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


def index_to_action(action_index: int) -> np.ndarray:
    action = np.zeros(3)  # steer, gas, brake
    action[1] = 0.1

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
        return Actions.BREAK.value
    return Actions.NOTHING.value


if __name__ == "__main__":
    train()

