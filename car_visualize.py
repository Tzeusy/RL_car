import numpy as np
import cv2
from PIL import Image

import gym
from pyglet.window import key

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time
import os
import random
from itertools import count
from enum import Enum

from model import DQN, ReplayMemory, Transition, Player
from plots import plot_rewards

import keras
import innvestigate
import scipy.misc

def innvestigate_input(model, input: np.ndarray):
    """
    :param model: Keras model
    :param input: 4-D numpy array of shape [n, h, w, c]
    """
    name = {
        0: 'lrp.sequential_preset_a_flat',
        1: 'guided_backprop',
        2: 'gradient',
    }[0]
    analyzer = innvestigate.create_analyzer(name, model)
    a = analyzer.analyze(input)

    # aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    return a


class Actions(Enum):
    NOTHING, GAS, BRAKE, LEFT, RIGHT = range(5)


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 30
MAX_EPISODE_LENGTH = 2000
LEARNING_RATE = 5e-3
REPLAY_MEM = 90000
snapshot_dir = "snapshots"

# Get number of actions from gym action space
n_actions = 5  # env.action_space.shape[0]

steps_done = 0
num_photos = 0
episode_rewards = []

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def create_env():
    env = gym.make('CarRacing-v0').unwrapped
    env.mode = 'fast'
    env.seed(0)
    return env


def get_screen(env, player=None):
    if player:
        env = player.env

    start = time.time()
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape

    if player:
        player.screen = screen

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    screen = resize(screen).unsqueeze(0).to(device)

    # print("{:.3f}s taken for get_screen".format(time.time() - start))
    return screen


def display_screens(players):
    start = time.time()

    full_screen = None

    for player in players:
        screen = player.screen #.env.render(mode='rgb_array') #.transpose((1, 0, 2))

        if full_screen is None:
            full_screen = screen
        else:
            border_width = 10
            height = full_screen.shape[1]
            border = np.zeros((3, height, border_width), dtype=np.uint8)

            full_screen = np.concatenate((full_screen, border, screen), axis=2)

    # TODO: Rewrite this to display in whatever UI library is being used
    full_screen = full_screen.transpose((1, 2, 0))
    full_screen = cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', full_screen)

    print("Time taken to render: {:.3f}s".format(time.time() - start))


def create_player(load_weights=True):
    env = create_env()
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen(env)
    _, n_channels, screen_height, screen_width = init_screen.shape # 3, 40, 60

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    model_dir = "semi_successful_models"
    model_file_name = "manual30_ep40.pth"
    policy_net.eval()
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.eval()

    if load_weights:
        policy_net.load_state_dict(torch.load(f"{model_dir}/{model_file_name}", map_location='cpu'))
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999999)
    memory = ReplayMemory(REPLAY_MEM)
    fake_memory = ReplayMemory(REPLAY_MEM)

    player = Player(env, policy_net, target_net, optimizer, scheduler, memory, fake_memory)
    return player

def select_action(player, state):
    # No EPS for visualizing
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        selected_action = player.policy_net(player.state).max(1)[1].item()

        if player.model_keras:
            start = time.time()
            global num_photos
            num_photos += 1
            save_interval = 1

            if num_photos % save_interval == 0:
                # player_state = player.state.cpu().numpy()
                output = innvestigate_input(player.model_keras, player_state)  # shape [n, h, w, c]

                # Normalize
                lrp_output = np.repeat(output[0][:,:,np.newaxis], repeats=3, axis=2)
                lrp_output -= np.min(lrp_output)
                lrp_output /= np.max(lrp_output)
                screen_output = player.screen_tensor[0,:].cpu().numpy().transpose(1, 2, 0)
                screen_output -= np.min(screen_output)
                screen_output /= np.max(screen_output)

                display = np.concatenate((lrp_output, screen_output), axis=1)
                display = Image.fromarray((display * 255).astype(np.uint8))
                display.save(f"output_{num_photos // save_interval}.png")
                # scipy.misc.imsave(f"output_{num_photos // save_interval}.png", display)
            print("Took {:.3f}s for lrp".format(time.time() - start))

    return selected_action

def step_player(player, player_done, fake_action):
    start = time.time()

    env = player.env
    real_action_idx = select_action(player, player.state)
    real_action = index_to_action(real_action_idx)

    fake_action_idx = action_to_index(fake_action)
    fake_action_available = fake_action_idx != Actions.NOTHING.value
    if fake_action_available:
        print("Inputting fake action for imitation")

    # if not fake_action_available:
    _, reward, done, _ = env.step(real_action)
    # else:
    #     _, reward, done, _ = env.step(fake_action)

    if(reward<0):
        player.consecutive_noreward += 1
    else:
        player.consecutive_noreward = 0

    if(player.consecutive_noreward > 50):
        if player.total_reward < 750:
            reward -= 100
        done = True

    if fake_action_available:
        if real_action_idx == fake_action_idx:
            reward += 5
        else:
            reward -= 5
    player.total_reward += reward

    # Observe new state
    last_screen = player.screen_tensor
    current_screen = get_screen(env, player)
    player.screen_tensor = current_screen

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
        player.fake_memory.push(player.state, fake_action_tensor, next_state, fake_reward)
    player.memory.push(player.state, real_action_tensor, next_state, reward)

    # Move to the next state
    player.state = next_state
    print("Step player took {:.3f}s".format(time.time() - start))

    return done

def train():
    os.makedirs(snapshot_dir, exist_ok=True)
    save_every = 20  # Save every 100 episodes
    display_interval = 2  # Display every 50 episodes
    num_episodes = 3000

    player1 = create_player()
    player2 = create_player(load_weights=False)
    player2.model_keras = None
    players = [player1]#, player2]

    fake_action = np.zeros(3)
    fake_action_listener(player1.env, fake_action)

    episode_rewards = []
    for i_episode in range(num_episodes):
        global steps_done
        steps_done += 1

        for player in players: # Execute for each player
            env = player.env
            # Initialize the environment and state
            env.seed(i_episode)
            env.reset()

        start = time.time()
        
        for player in players:
            env = player.env
            last_screen = get_screen(env, player)
            current_screen = get_screen(env, player)
            player.state = current_screen - last_screen
            player.screen_tensor = current_screen

            player.total_reward = 0  # TODO: Update to be playerwise
            player.consecutive_noreward = 0
            # action_barchart = np.zeros(n_actions)

        # Keeps track of which players are done with the current episode
        player_done = [False for p in players]
        for t in count():
            if all(player_done):
                break

            for player_i, player in enumerate(players):
                if player_done[player_i]:
                    continue
                done = step_player(player, player_done, fake_action)

                if (done or t > MAX_EPISODE_LENGTH):
                    player.state = None
                    print(f"Episode {i_episode} with {t} length took {time.time()-start}s "
                          f"and scored {player.total_reward}")

                    player_done[player_i] = True

            # if i_episode % display_interval == 0: # or i_episode < 100:
            #     display_screens(players)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            for player in players:
                player.target_net.load_state_dict(player.policy_net.state_dict())

        current_reward = np.mean(episode_rewards[-100:-1])

        # Save for user
        if i_episode % save_every == 0 and i_episode > 0:
            filename = f'{snapshot_dir}/target_episode{i_episode}_reward_{current_reward:.3f}.pth'
            torch.save(player1.target_net.state_dict(), filename)

    print('Complete')
    for player in players:
        player.env.close()


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
        return Actions.BRAKE.value
    return Actions.NOTHING.value


if __name__ == "__main__":
    train()

