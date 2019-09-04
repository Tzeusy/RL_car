from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton


class StartButton(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.button = QPushButton('VROOM VROOM')
        self.button.setDefault(True)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)
