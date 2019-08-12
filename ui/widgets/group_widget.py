from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox


class GroupWidget(QWidget):
    def __init__(self, widgets, title='', parent=None):
        super().__init__(parent)
        self.widgets = widgets
        self.title = title

        layout = QVBoxLayout()
        layout.addWidget(self.create_group_box())
        self.setLayout(layout)

    def create_group_box(self):
        layout = QVBoxLayout()
        for widget in self.widgets.values():
            layout.addWidget(widget)

        group_box = QGroupBox(self.title)
        group_box.setLayout(layout)
        return group_box
