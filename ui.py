from PyQt5.QtWidgets import QApplication
from ui import Main

import car


def app():
    qapp = QApplication([])
    main_widget = Main()
    main_widget.show()
    qapp.exec_()

    params = main_widget.hyperparameters
    print(params)

    car.EPS_START = params.eps[0]
    car.EPS_END = params.eps[1]
    car.LEARNING_RATE = params.lr
    car.REPLAY_MEM = params.memory_size
    car.IMITATION_REWARD = params.imitation_reward
    car.KERNEL_SIZE = params.ksize
    car.N_LAYERS = params.n_layers

    print('VROOM VROOM')
    car.train()


if __name__ == '__main__':
    app()
