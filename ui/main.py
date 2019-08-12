import sys
from collections import namedtuple
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from .widgets import (GroupWidget,
                     LrSlider,
                     ArchiSlider,
                     EpsSlider,
                     MemorySlider,
                     ImitationSlider,
                     StartButton)


class Main(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.nn_widgets = {
            'lr': LrSlider(),
            'archi': ArchiSlider(),
        }
        self.rl_widgets = {
            'eps': EpsSlider(),
            'memory': MemorySlider(),
            'imitation': ImitationSlider(),
        }
        self.hyperparameters = self.get_hyperparameters()

        self.button = StartButton()
        self.button.button.clicked.connect(self.onclick)

        layout = QVBoxLayout()
        layout.addWidget(self.create_content_widget())
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.setWindowTitle('Hyperparameters')

    def create_content_widget(self):
        layout = QHBoxLayout()
        layout.addWidget(GroupWidget(widgets=self.nn_widgets,
                                     title='Neural network parameters'))
        layout.addWidget(GroupWidget(widgets=self.rl_widgets,
                                     title='Reinforcement learning parameters'))

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def get_hyperparameters(self):
        lr = self.nn_widgets['lr'].get_value()
        ksize, n_layers = self.nn_widgets['archi'].get_value()
        eps = self.rl_widgets['eps'].get_value()
        memory_size = self.rl_widgets['memory'].get_value()
        imitation_reward = self.rl_widgets['imitation'].get_value()

        Hyperparameters = namedtuple('Hyperparameters', 'lr '
                                                        'ksize '
                                                        'n_layers '
                                                        'eps '
                                                        'memory_size '
                                                        'imitation_reward')
        return Hyperparameters(lr=lr, ksize=ksize, n_layers=n_layers, eps=eps,
                               memory_size=memory_size, imitation_reward=imitation_reward)

    def onclick(self):
        self.hyperparameters = self.get_hyperparameters()
        self.close()


def main():
    app = QApplication(sys.argv)
    main_widget = Main()
    main_widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
