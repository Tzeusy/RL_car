from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel


class LabelledSlider(QWidget):
    def __init__(self, text, slider, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        layout.addWidget(QLabel(text))
        layout.addWidget(slider)
        self.setLayout(layout)
