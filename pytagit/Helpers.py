import numpy as np
from random import Random
from functools import partial
from PyQt6 import QtWidgets, QtGui, QtCore
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from .CNNTrainer import CNNTrainer, ImageDataset, transform_fun
import albumentations as A
from PIL import Image
from einops import rearrange
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from tqdm.rich import tqdm
from sklearn.metrics import pairwise_distances



class VisualThresholdSelector(QtWidgets.QDialog):
    def __init__(self, feature_vectors, clean_features, image_paths, distance_function='cosine', use_lad=True, num_bins=10, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Threshold Selector")
        self.setModal(True)

        # Parameters
        self.feature_vectors = feature_vectors
        self.clean_features = clean_features
        self.image_paths = image_paths  # Add image_paths as an attribute
        self.distance_function = distance_function
        self.use_lad = use_lad  # Initialize the use_lad attribute
        self.num_bins = num_bins  # Number of distance bins

        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Select the rightmost image that you consider part of the cluster.\n"
            "Images are ordered by their distance from the clean distribution."
        )
        layout.addWidget(instructions)

        # Image Grid
        self.image_grid = QtWidgets.QGridLayout()
        self.representative_images = self.calculate_representative_images()
        self.distance_labels = []  # Store references to distance labels
        self.add_images_to_grid()
        layout.addLayout(self.image_grid)

        # Selected Threshold
        self.selected_threshold = None

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button = QtWidgets.QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setDefault(True)
        self.cancel_button.setFixedHeight(50)
        self.confirm_button.setFixedHeight(50)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)
        layout.addLayout(button_layout)

    def calculate_representative_images(self):
        """
        Calculate representative images for each distance bin.
        :return: List of tuples (distance, representative_image_path).
        """
        # Compute pairwise distances
        distances = pairwise_distances(
            self.feature_vectors.numpy(), 
            self.clean_features.numpy(), 
            metric=self.distance_function
        )
        avg_distances = np.mean(distances, axis=1) if self.use_lad else np.zeros(distances.shape[0])

        # Bin distances
        min_dist = np.min(avg_distances)
        max_dist = np.percentile(avg_distances, 20)  # Use the 20th percentile as the upper bound
        bins = np.linspace(min_dist, max_dist, self.num_bins + 1)
        binned_indices = np.digitize(avg_distances, bins) - 1

        # Select representative images for each bin
        representative_images = []
        for bin_idx in range(self.num_bins):
            indices_in_bin = np.where(binned_indices == bin_idx)[0]
            if len(indices_in_bin) > 0:
                median_index = indices_in_bin[len(indices_in_bin) // 2]  # Median index (deterministic)
                representative_images.append((bins[bin_idx], self.image_paths[median_index]))
        return representative_images

    def add_images_to_grid(self):
        """Add representative images to the grid."""
        for col, (distance, image_path) in enumerate(self.representative_images):
            # Load and display image
            pixmap = QtGui.QPixmap(image_path).scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setToolTip(f"Distance: {distance:.3f}")
            label.mousePressEvent = partial(self.select_image, distance=distance, column=col)
            self.image_grid.addWidget(label, 0, col)

            # Add distance label
            distance_label = QtWidgets.QLabel(f"{distance:.3f}")
            distance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            distance_label.setStyleSheet("background-color: white; padding: 5px;")  # Default style
            self.image_grid.addWidget(distance_label, 1, col)
            self.distance_labels.append(distance_label)  # Store reference to the label

    def select_image(self, event, distance, column):
        """Handle image selection."""
        self.selected_threshold = distance

        # Reset all labels to default style
        for label in self.distance_labels:
            label.setStyleSheet("background-color: white; padding: 5px;")

        # Highlight the selected label
        selected_label = self.distance_labels[column]
        selected_label.setStyleSheet("background-color: lightblue; padding: 5px; font-weight: bold;")

    def get_threshold(self):
        """Return the selected threshold."""
        return self.selected_threshold


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

        # accept threshold input
        self.accept_threshold_input = QtWidgets.QDoubleSpinBox()
        self.accept_threshold_input.setRange(1e-6, 1.0)
        self.accept_threshold_input.setDecimals(2)
        self.accept_threshold_input.setSingleStep(0.01)
        self.accept_threshold_input.setValue(0.90)  # Default value
        layout.addWidget(QtWidgets.QLabel("Accept threshold:"))
        layout.addWidget(self.accept_threshold_input)

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
            "threshold": self.accept_threshold_input.value(),
            "n_estimators": self.n_estimators_input.value(),
            "max_depth": self.max_depth_input.value(),
        }
    

class kNNDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("kNN Parameters")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        # missing:
        # algorithm='auto',
        # metric= 'minkowski'

        # accept threshold input
        self.accept_threshold_input = QtWidgets.QDoubleSpinBox()
        self.accept_threshold_input.setRange(1e-6, 1.0)
        self.accept_threshold_input.setDecimals(2)
        self.accept_threshold_input.setSingleStep(0.01)
        self.accept_threshold_input.setValue(0.90)  # Default value
        layout.addWidget(QtWidgets.QLabel("Accept threshold:"))
        layout.addWidget(self.accept_threshold_input)

        # Number of neighbours input
        self.neighbours_input = QtWidgets.QSpinBox()
        self.neighbours_input.setRange(1, 100)
        self.neighbours_input.setValue(3)  # Default value
        layout.addWidget(QtWidgets.QLabel("Number of Neighbours:"))
        layout.addWidget(self.neighbours_input)

        # leaf_size input
        self.leaf_size_input = QtWidgets.QSpinBox()
        self.leaf_size_input.setRange(1, 100)
        self.leaf_size_input.setValue(10)  # Default value
        layout.addWidget(QtWidgets.QLabel("Leaf Size:"))
        layout.addWidget(self.leaf_size_input)

        # p input
        self.p_input = QtWidgets.QSpinBox()
        self.p_input.setRange(1, 3)
        self.p_input.setValue(2)  # Default value
        layout.addWidget(QtWidgets.QLabel("P:"))
        layout.addWidget(self.p_input)

        # Run button
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.accept)
        layout.addWidget(self.run_button)

    def get_parameters(self):
        return {
            "threshold": self.accept_threshold_input.value(),
            "neighbours": self.neighbours_input.value(),
            "leaf_size": self.leaf_size_input.value(),
            "p": self.p_input.value(),
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
        self.batch_size_train_input = QtWidgets.QSpinBox()
        self.batch_size_train_input.setRange(1, 1024)
        self.batch_size_train_input.setValue(32)  # Default value
        layout.addWidget(QtWidgets.QLabel("Batch Size (train):"))
        layout.addWidget(self.batch_size_train_input)

        # Batch size input
        self.batch_size_test_input = QtWidgets.QSpinBox()
        self.batch_size_test_input.setRange(1, 1024)
        self.batch_size_test_input.setValue(64)  # Default value
        layout.addWidget(QtWidgets.QLabel("Batch Size (test):"))
        layout.addWidget(self.batch_size_test_input)

        # Num workers
        self.num_workers_input = QtWidgets.QSpinBox()
        self.num_workers_input.setRange(1, 30)
        self.num_workers_input.setValue(8)  # Default value
        layout.addWidget(QtWidgets.QLabel("Num Workers:"))
        layout.addWidget(self.num_workers_input)

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
            "batch_size_train": self.batch_size_train_input.value(),
            "batch_size_test": self.batch_size_test_input.value(),
            "num_workers": self.num_workers_input.value(),
            "pretrained": self.pretrained_checkbox.isChecked()
        }
    



class MultiClassClassifier(object):

    def get_name(self):
        raise ValueError('You must overload the class.')

    def get_dialog(self, parent):
        raise ValueError('You must overload the class.')
    
    def train(self, params, X_train, y_train, filenames, id_undefined_class, num_classes):
        raise ValueError('You must overload the class.')
    
    def classify(self, params, filenames, features, id_undefined_class):
        raise ValueError('You must overload the class.')
    
    


class RFClassifier(MultiClassClassifier):

    def get_name(self):
        return 'Random Forest'

    def get_dialog(self, parent=None):
        return RandomForestDialog(parent)
    

    def train(self, params, X_train, y_train, filenames, id_undefined_class, num_classes):
        # Train Random Forest classifier
        self.clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        self.clf.fit(X_train, y_train)
        return True


    def classify(self, params, filenames, features, id_undefined_class):
        # perform classification
        # predictions = self.clf.predict(features.numpy())
        # compute probabilities
        probs = self.clf.predict_proba(features.numpy())
        # get predictions
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        # reset ones with low probability
        # predictions[probs.max(axis=-1) < params['threshold']] = id_undefined_class
        # return them
        return predictions, confidences
    



class kNNClassifier(MultiClassClassifier):

    def get_name(self):
        return 'kNN'

    def get_dialog(self, parent=None):
        return kNNDialog(parent)
    

    def train(self, params, X_train, y_train, filenames, id_undefined_class, num_classes):
        self.knn = KNeighborsClassifier(
            n_neighbors = params["neighbours"],
            algorithm='auto',
            leaf_size = params["leaf_size"],
            p = params["p"],
            metric= 'minkowski'
        )
        self.knn.fit(X_train, y_train)
        return True


    def classify(self, params, filenames, features, id_undefined_class):
        # perform classification
        # predictions = self.clf.predict(features.numpy())
        # compute probabilities
        probs = self.knn.predict_proba(features.numpy())
        # get predictions
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        # return them
        return predictions, confidences
    



class CNNClassifier(MultiClassClassifier):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.model = None


    def get_name(self):
        return 'CNN'

    def get_dialog(self, parent=None):
        return CNNTrainingDialog(parent)
    

    def train(self, params, X_train, y_train, filenames, id_undefined_class, num_classes):
        
        # define transformations and dataloader
        transform = partial(transform_fun, train=True, sz=256)
        dataset = ImageDataset(filenames, y_train, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size_train'], shuffle=True, drop_last=False, num_workers=params['num_workers'])
        
        # define model
        self.model = CNNTrainer(params['model'], params['learning_rate'], params['batch_size_train'], params['pretrained'], num_classes)
        
        trainer = pl.Trainer(
            max_epochs=params['epochs'],
            callbacks=[RichProgressBar(), ModelCheckpoint(monitor='train_loss', mode='min')],
            accelerator="auto"
        )
        
        trainer.fit(self.model, dataloader)

        # return true as training has been computed
        return True


    
    def classify(self, params, filenames, features, id_undefined_class):
        # define transforms
        transform = partial(transform_fun, train=False, sz=256)
        # init model and output
        self.model.eval()
        predictions = []
        probs = []
        # create dataloader
        dataset = ImageDataset(filenames, None, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size_test'], shuffle=False, drop_last=False, num_workers=params['num_workers'])

        for batch in tqdm(dataloader):
            images, labels = batch
            with torch.no_grad():
                output = self.model(images)
                output = F.softmax(output, dim=0)
                prob, pred_label = output.max(dim=1)
                pred_label[prob < params['threshold']] = id_undefined_class
                predictions.append(pred_label)
                probs.append(prob)

        predictions = torch.cat(predictions)
        probs = torch.cat(probs)

        
        return predictions, probs
    




if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    # define classifiers
    classifiers = [RFClassifier(), CNNClassifier(), kNNClassifier()]

    for cur_classifier in classifiers:
        # get dialog
        dialog = cur_classifier.get_dialog()
        # if accepted
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # get parameters
            params = dialog.get_parameters()
            # print them
            print(params)

