from PyQt5.QtWidgets import (QVBoxLayout, QListWidget,
                             QWidget, QSlider, QLabel)
from PyQt5.QtCore import Qt
from .labelled_slider import LabelledSlider


class ArchiSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kernel_size = 3
        self.nn_depth = 4

        self.list_widget = QListWidget()
        self.format_list_widget()

        self.slider_kernel = self.create_slider_kernel()
        self.slider_depth = self.create_slider_depth()

        layout = QVBoxLayout()
        layout.addWidget(
            LabelledSlider(text='Kernel size:', slider=self.slider_kernel)
        )
        layout.addWidget(
            LabelledSlider(text='No. of conv layers:', slider=self.slider_depth)
        )
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def get_value(self):
        return self.kernel_size, self.nn_depth

    def slider_change(self):
        self.kernel_size = self.slider_kernel.value()
        self.nn_depth = self.slider_depth.value()
        self.format_list_widget()

    def format_list_widget(self):
        layers = []
        ksize = self.kernel_size
        for i in range(self.nn_depth):
            n_filters = 16 * 2**i
            layers.append(f'conv{i+1}: nn.Conv2d(3, {n_filters}, kernel_size={ksize}, stride=2)')
            layers.append(f'  bn{i+1}: nn.BatchNorm2d({n_filters})')
            layers.append('')

        self.list_widget.clear()
        self.list_widget.addItems(layers)

    def create_slider_kernel(self):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(2)
        slider.setMaximum(7)
        slider.setValue(self.kernel_size)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.slider_change)
        return slider

    def create_slider_depth(self):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(7)
        slider.setValue(self.nn_depth)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.slider_change)
        return slider
