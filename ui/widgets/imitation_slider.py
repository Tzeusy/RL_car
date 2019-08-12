from PyQt5.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt


class ImitationSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.imitation_reward = 5

        self.label = QLabel(self.format_imitation_reward())

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(20)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setValue(self.imitation_reward)
        slider.valueChanged.connect(self.slider_change)
        self.slider = slider

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def get_value(self):
        return self.imitation_reward

    def slider_change(self):
        self.imitation_reward = self.slider.value()
        self.label.setText(self.format_imitation_reward())

    def format_imitation_reward(self):
        return f'Imitation reward: ±{self.imitation_reward}'
