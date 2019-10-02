from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QCheckBox


class StartButton(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.button = QPushButton('VROOM VROOM')
        self.button.setDefault(True)

        self.checkbox = QCheckBox('Freeze weights?')

        layout = QGridLayout()
        layout.addWidget(self.button, 1, 1, 1, 3)
        layout.addWidget(self.checkbox, 1, 4, 1, 1)
        self.setLayout(layout)
