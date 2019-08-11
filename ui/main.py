import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from widgets import (GroupWidget,
                     LrSlider,
                     ArchiSlider,
                     EpsSlider,
                     MemorySlider,
                     ImitationSlider,
                     StartButton)

# eps, lr, neural network width/depth, replay_mem [for ddqn], imitation reward (?)


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

    def onclick(self):
        lr = self.nn_widgets['lr'].get_value()
        print('LR:', lr)

        ksize, n_layers = self.nn_widgets['archi'].get_value()
        print('Kernel size:', ksize)
        print('No. of conv layers:', n_layers)

        eps = self.rl_widgets['eps'].get_value()
        print('Eps:', eps)

        memory = self.rl_widgets['memory'].get_value()
        print('Memory size:', memory)

        imitation_reward = self.rl_widgets['imitation'].get_value()
        print('Imitation reward:', imitation_reward)


def main():
    app = QApplication(sys.argv)
    main_widget = Main()
    main_widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
