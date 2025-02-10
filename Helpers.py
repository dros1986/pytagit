from PyQt6 import QtWidgets, QtGui, QtCore



class ThresholdDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Threshold")
        self.setModal(True)
        self.threshold = 0.02  # Default threshold

        layout = QtWidgets.QVBoxLayout(self)

        # Threshold input
        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(self.threshold)
        self.threshold_input.setDecimals(3)
        layout.addWidget(QtWidgets.QLabel("Threshold:"))
        layout.addWidget(self.threshold_input)

        # Run button
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.accept)
        layout.addWidget(self.run_button)

    def get_threshold(self):
        return self.threshold_input.value()


class RandomForestDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Forest Parameters")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        # Number of trees input
        self.n_estimators_input = QtWidgets.QSpinBox()
        self.n_estimators_input.setRange(1, 1000)
        self.n_estimators_input.setValue(100)  # Default value
        layout.addWidget(QtWidgets.QLabel("Number of Trees:"))
        layout.addWidget(self.n_estimators_input)

        # Max depth input
        self.max_depth_input = QtWidgets.QSpinBox()
        self.max_depth_input.setRange(1, 100)
        self.max_depth_input.setValue(10)  # Default value
        layout.addWidget(QtWidgets.QLabel("Max Depth:"))
        layout.addWidget(self.max_depth_input)

        # Run button
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.accept)
        layout.addWidget(self.run_button)

    def get_parameters(self):
        return {
            "n_estimators": self.n_estimators_input.value(),
            "max_depth": self.max_depth_input.value(),
        }
    

class CNNTrainingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CNN Training Parameters")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        # Model selection dropdown
        self.model_dropdown = QtWidgets.QComboBox()
        self.model_dropdown.addItems(["ResNet18", "ResNet34", "ResNet50"])
        layout.addWidget(QtWidgets.QLabel("Select CNN Architecture:"))
        layout.addWidget(self.model_dropdown)

        # accept threshold input
        self.accept_threshold_input = QtWidgets.QDoubleSpinBox()
        self.accept_threshold_input.setRange(1e-6, 1.0)
        self.accept_threshold_input.setDecimals(2)
        self.accept_threshold_input.setSingleStep(0.01)
        self.accept_threshold_input.setValue(0.90)  # Default value
        layout.addWidget(QtWidgets.QLabel("Accept threshold:"))
        layout.addWidget(self.accept_threshold_input)

        # Epochs input
        self.nepochs_input = QtWidgets.QSpinBox()
        self.nepochs_input.setRange(1, 1000)
        self.nepochs_input.setValue(20)  # Default value
        layout.addWidget(QtWidgets.QLabel("Epochs:"))
        layout.addWidget(self.nepochs_input)

        # Learning rate input
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setRange(1e-6, 1.0)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setValue(0.001)  # Default value
        layout.addWidget(QtWidgets.QLabel("Learning Rate:"))
        layout.addWidget(self.learning_rate_input)

        # Batch size input
        self.batch_size_input = QtWidgets.QSpinBox()
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(32)  # Default value
        layout.addWidget(QtWidgets.QLabel("Batch Size:"))
        layout.addWidget(self.batch_size_input)

        # Pretrained checkbox
        self.pretrained_checkbox = QtWidgets.QCheckBox("Use Pretrained Weights")
        self.pretrained_checkbox.setChecked(True)  # Default to checked
        layout.addWidget(self.pretrained_checkbox)

        # Run button
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.accept)
        layout.addWidget(self.run_button)

    def get_parameters(self):
        return {
            "model": self.model_dropdown.currentText(),
            "threshold": self.accept_threshold_input.value(),
            "epochs": self.nepochs_input.value(),
            "learning_rate": self.learning_rate_input.value(),
            "batch_size": self.batch_size_input.value(),
            "pretrained": self.pretrained_checkbox.isChecked()
        }