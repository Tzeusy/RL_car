from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout,
                             QWidget, QSlider, QLabel)
from PyQt5.QtCore import Qt
from .labelled_slider import LabelledSlider


class EpsSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.eps_times_100 = (20, 5)  # 0.20 -> 0.05

        self.label = QLabel(self.format_eps())
        self.slider_start = self.create_slider_start()
        self.slider_end = self.create_slider_end()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(
            LabelledSlider(text='Start:', slider=self.slider_start)
        )
        layout.addWidget(
            LabelledSlider(text='  End:', slider=self.slider_end)
        )
        self.setLayout(layout)

    def get_value(self):
        eps_start, eps_end = self.eps_times_100[0]/100, self.eps_times_100[1]/100
        return eps_start, eps_end

    def slider_change(self):
        start_value = self.slider_start.value()
        end_value = min(start_value, self.slider_end.value())
        self.slider_end.setValue(end_value)

        self.eps_times_100 = (start_value, end_value)
        self.label.setText(self.format_eps())

    def format_eps(self):
        eps_start, eps_end = self.get_value()
        return f'Epsilon: {eps_start:.2f} -> {eps_end:.2f}'

    def create_slider_start(self):
        slider_start = QSlider(Qt.Horizontal)
        slider_start.setMinimum(0)
        slider_start.setMaximum(100)
        slider_start.setValue(self.eps_times_100[0])
        slider_start.valueChanged.connect(self.slider_change)
        return slider_start

    def create_slider_end(self):
        slider_end = QSlider(Qt.Horizontal)
        slider_end.setMinimum(0)
        slider_end.setMaximum(100)
        slider_end.setValue(self.eps_times_100[1])
        slider_end.valueChanged.connect(self.slider_change)
        return slider_end

