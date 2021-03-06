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

from model import DQN, DQNUser, ReplayMemory, Transition, Player


def innvestigate_input(analyzer, input: np.ndarray):
    """
    :param model: Keras model
    :param input: 4-D numpy array of shape [n, h, w, c]
    """
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
IMITATION_REWARD = 5
KERNEL_SIZE = 3
N_LAYERS = 4
snapshot_dir = "snapshots"

# Get number of actions from gym action space
n_actions = 5  # env.action_space.shape[0]

steps_done = 0
num_photos = 0
do_optimize = False

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

robot_image = np.array(Image.open("images/robot.png"))


def create_env():
    env = gym.make('CarRacing-v0').unwrapped
    env.mode = 'fast'
    env.seed(0)
    return env


def get_screen(env, player=None):
    if player:
        env = player.env

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
    return screen


def display_screens(players, i_episode):
    full_screen = None

    for i, player in enumerate(players):
        screen = player.screen
        channels, height, width = screen.shape
        border = np.zeros((3, height, 10), dtype=np.uint8)  # black border for divider

        lrp_output = cv2.resize(player.lrp_output, dsize=(width, height),
                                interpolation=cv2.INTER_CUBIC)  # maybe INTER_NEAREST instead?
        # Repeat to make RGB channels
        lrp_output = np.repeat(lrp_output[:, :, np.newaxis], repeats=3, axis=2)
        lrp_output = lrp_output.transpose((2, 0, 1))

        # Normalize
        lrp_output -= lrp_output.min()
        lrp_output /= lrp_output.max()
        # lrp_output[lrp_output < 0] = 0

        lrp_output = (lrp_output * 255).astype(np.uint8)

        screen = np.concatenate((screen, border, lrp_output), axis=2)

        # Add robot
        if i == 1:
            robot_h, robot_w, c = robot_image.shape
            screen[:, :robot_h, :robot_w,] = robot_image[:,:,:3].transpose((2, 0, 1))  # Omit A channel

        if full_screen is None:
            full_screen = screen
        else:
            full_screen = np.concatenate((full_screen, screen), axis=1)

    full_screen = full_screen.transpose((1, 2, 0))
    full_screen = cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(full_screen, f'Episode {i_episode+1}', (10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)

    text = 'Training model...' if do_optimize else 'Weights frozen!'
    cv2.putText(full_screen, text, (300, 30), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    name = 'VROOM VROOM'
    cv2.namedWindow(name)
    cv2.moveWindow(name, 100, 50)
    cv2.imshow(name, full_screen)
    cv2.waitKey(1)


def create_player(load_weights=True, user_model=False):
    env = create_env()
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_screen(env)
    _, n_channels, screen_height, screen_width = init_screen.shape # 3, 40, 60

    if user_model:
        policy_net = DQNUser(screen_height, screen_width, n_actions,
                             KERNEL_SIZE, N_LAYERS).to(device)
        policy_net.eval()
        target_net = DQNUser(screen_height, screen_width, n_actions,
                             KERNEL_SIZE, N_LAYERS).to(device)
        target_net.eval()
    else:
        policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        policy_net.eval()
        target_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net.eval()

    if load_weights:
        model_dir = "models"
        model_file_name = "mean100_659.pth"
        policy_net.load_state_dict(torch.load(f"{model_dir}/{model_file_name}", map_location='cpu'))
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999999)
    memory = ReplayMemory(REPLAY_MEM)
    fake_memory = ReplayMemory(REPLAY_MEM)

    player = Player(env, policy_net, target_net, optimizer, scheduler, memory, fake_memory)
    return player


def select_action(player):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * steps_done / EPS_DECAY)

    # selected_action = random.randint(0, n_actions - 1)
    # Nothing, Forward, Brake, Left, Right
    selected_action = random.choices(range(5), [0.05, 0.45, 0.002, 0.24, 0.24])[0]

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = player.policy_net(player.state).max(1)[1].item()

    if player.model_keras:
        global num_photos
        num_photos += 1
        save_interval = 1

        if num_photos % save_interval == 0:
            player_state = player.state.cpu().numpy()
            output = innvestigate_input(player.analyzer, player_state)  # shape [n, h, w, c]

            player.lrp_output = output[0]

    return selected_action


def optimize_model(player):
    if len(player.memory) < BATCH_SIZE or len(player.fake_memory) < BATCH_SIZE:
        return
    if len(player.fake_memory) >= BATCH_SIZE:
        transitions = player.memory.sample(BATCH_SIZE // 2) + player.fake_memory.sample(BATCH_SIZE // 2)
    else:
        transitions = player.memory.sample(BATCH_SIZE)
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
    state_action_values = player.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    max_idx = player.policy_net(non_final_next_states).max(1)[1]

    # DDQN
    next_state_values[non_final_mask] = torch.gather(player.target_net(non_final_next_states), dim=1, index=max_idx[:, None]).squeeze()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    player.optimizer.zero_grad()
    loss.backward()
    for param in player.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    player.optimizer.step()
    player.scheduler.step()


def step_player(player, fake_action):
    env = player.env
    real_action_idx = select_action(player)
    real_action = index_to_action(real_action_idx)

    fake_action_idx = action_to_index(fake_action)
    fake_action_available = fake_action_idx != Actions.NOTHING.value
    if fake_action_available:
        print("Inputting fake action for imitation:", Actions(fake_action_idx))
        _, reward, done, _ = env.step(fake_action)
    else:
        _, reward, done, _ = env.step(real_action)

    if reward < 0:
        player.consecutive_noreward += 1
    else:
        player.consecutive_noreward = 0

    if player.consecutive_noreward > 50:
        if player.total_reward < 750:
            reward -= 100
        done = True

    if real_action_idx == fake_action_idx:
        reward += IMITATION_REWARD
    else:
        reward -= IMITATION_REWARD
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

    if do_optimize:
        optimize_model(player)

    return done

def train():
    os.makedirs(snapshot_dir, exist_ok=True)
    player1 = create_player(load_weights=True, user_model=False)
    player2 = create_player(load_weights=False)
    players = [player1, player2]

    fake_action = np.zeros(3)
    fake_action_listener(player1.env, fake_action)

    save_every = 100  # Save every 100 episodes
    display_interval = 1  # Display every 2 steps
    num_episodes = 3000

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

        # Keeps track of which players are done with the current episode
        player_done = [False for p in players]
        for t in count():
            if all(player_done):
                break

            for player_i, player in enumerate(players):
                if player_done[player_i]:
                    continue
                done = step_player(player, fake_action)

                if done or t > MAX_EPISODE_LENGTH:
                    player.state = None
                    print(f"Episode {i_episode} with {t} length took {time.time()-start}s "
                          f"and scored {player.total_reward}")

                    player_done[player_i] = True

            if t % display_interval == 0:  # or i_episode < 100:
                display_screens(players, i_episode)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            for player in players:
                player.target_net.load_state_dict(player.policy_net.state_dict())

        # Save for user
        if i_episode % save_every == 0 and i_episode > 0:
            filename = f'{snapshot_dir}/target_episode{i_episode}.pth'
            torch.save(player1.target_net.state_dict(), filename)
    
    print('Complete')

    for player in players:
        player.env.close()


def fake_action_listener(env, fake_action):
    def key_press(k, mod):
        # print(k)
        if k == key.LEFT:
            fake_action[0] = -1.0
        elif k == key.RIGHT:
            fake_action[0] = 1.0
        elif k == key.UP:
            fake_action[1] = 1.0
        elif k == key.DOWN:
            fake_action[2] = 0.8  # set 1.0 for wheels to block to zero rotation
        elif k == key.SPACE:
            global do_optimize
            do_optimize = not do_optimize

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

