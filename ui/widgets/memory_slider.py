from PyQt5.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt


class MemorySlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memory_size = 90_000
        self.step_size = 5_000

        self.label = QLabel(self.format_memory_size())

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(200_000 // self.step_size)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
        slider.setValue(self.memory_size // self.step_size)
        slider.valueChanged.connect(self.slider_change)
        self.slider = slider

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def get_value(self):
        return self.memory_size

    def slider_change(self):
        self.memory_size = self.slider.value() * self.step_size
        self.label.setText(self.format_memory_size())

    def format_memory_size(self):
        return f'Replay memory size: {self.memory_size:,}'
