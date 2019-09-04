import numpy as np
import matplotlib.pyplot as plt
import torch
import keras
import innvestigate
from pytorch2keras.converter import pytorch_to_keras
from tqdm import trange
from car import create_env, get_screen as get_train_screen
from model import DqnNoFc, DQN

import scipy.misc

keras.backend.set_image_data_format('channels_first')


def main():
    model_path = './semi_successful_models/manual30_ep40.pth'
    model = create_model(model_path)

    env = create_env()
    env.reset()
    last_screen = get_screen(env)

    # take 300 random actions first (arbitrary)
    for _ in trange(300):
        random_action = env.action_space.sample()
        env.step(random_action)
        screen = get_screen(env)
        last_screen = screen - last_screen

    output = innvestigate_input(model, last_screen)  # shape [n, h, w, c]
    env.close()

    output_norm = output[0]
    # output_norm /= np.max(output_norm)
    # print(np.min(output_norm), np.max(output_norm))

    scipy.misc.imsave("screen.png", (screen-last_screen)[0].transpose([1, 2, 0]))
    scipy.misc.imsave("screen_output.png", output_norm)

    # plt.figure(1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(screen[0].transpose([1, 2, 0]))
    # plt.subplot(1, 2, 2)
    # plt.imshow(output[0], cmap='seismic', clim=(-1, 1))
    # plt.show(block=True)


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


def create_model(model_path: str=None):
    env = create_env()
    env.reset()

    _, n_channels, screen_height, screen_width = get_screen(env).shape
    model = get_torch_model(screen_height, screen_width, model_path)
    model_keras = torch_to_keras(model, image_shape=[n_channels, screen_height, screen_width])

    env.close()
    return model_keras


def get_torch_model(screen_height, screen_width, model_path):
    model = DQN(screen_height, screen_width, 5)

    if model_path is not None:
        state_dict = torch.load(model_path, map_location='cpu')
        # state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict)

    model.eval()
    return model


def torch_to_keras(model, image_shape):
    """
    :param model: instance of PyTorch model
    :param image_shape: list of [c, h, w]
    """
    # use dummy variable to trace the model (see github README)
    input_np = np.random.uniform(0, 1, [1, *image_shape])  # add batch dimension
    input_var = torch.autograd.Variable(torch.FloatTensor(input_np))

    input_shapes = [image_shape]
    return pytorch_to_keras(model, input_var, input_shapes, verbose=False)


def get_screen(env) -> np.ndarray:
    # use this to obtain same screen used in car.py
    return get_train_screen(env).cpu().numpy()

    # use this for original screen without resizing
    screen = env.render(mode='rgb_array').transpose([2, 0, 1])
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = np.expand_dims(screen, axis=0)
    return screen


if __name__ == '__main__':
    main()
