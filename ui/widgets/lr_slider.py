from PyQt5.QtWidgets import (QVBoxLayout,
                             QWidget, QSlider, QLabel)
from PyQt5.QtCore import Qt
from .labelled_slider import LabelledSlider


class LrSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lr = (5, 3)  # 5e-3

        self.label = QLabel(self.format_lr())
        self.slider_coef = self.create_slider_coef()
        self.slider_exp = self.create_slider_exp()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(
            LabelledSlider(text='Coefficient:', slider=self.slider_coef)
        )
        layout.addWidget(
            LabelledSlider(text='Exponent:', slider=self.slider_exp)
        )
        self.setLayout(layout)

    def get_value(self):
        coef, exp = self.lr
        return coef * 10**(-exp)

    def slider_change(self):
        coef_value = self.slider_coef.value()
        exp_value = self.slider_exp.value()
        self.lr = (coef_value, exp_value)
        self.label.setText(self.format_lr())

    def format_lr(self):
        value = self.get_value()
        return f'LR: {value:.1e}'

    def create_slider_coef(self):
        slider_coef = QSlider(Qt.Horizontal)
        slider_coef.setMinimum(0)
        slider_coef.setMaximum(9)
        slider_coef.setValue(self.lr[0])
        slider_coef.setTickPosition(QSlider.TicksBelow)
        slider_coef.valueChanged.connect(self.slider_change)
        return slider_coef

    def create_slider_exp(self):
        slider_exp = QSlider(Qt.Horizontal)
        slider_exp.setMinimum(1)
        slider_exp.setMaximum(6)
        slider_exp.setValue(self.lr[1])
        slider_exp.setTickPosition(QSlider.TicksBelow)
        slider_exp.setTickInterval(1)
        slider_exp.valueChanged.connect(self.slider_change)
        return slider_exp
