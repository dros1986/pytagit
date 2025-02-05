import os
import sys
import json
import torch
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from sklearn import preprocessing
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tqdm import tqdm
from PIL import Image
import albumentations as A
import torch.nn.functional as F
from einops import rearrange
import cv2
from OOD4Inclusion import OOD4Inclusion  # Import the OOD4Inclusion class
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier


class DraggableLabel(QtWidgets.QLabel):
    def __init__(self, image_path, main_window, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.main_window = main_window  # Reference to the main application window
        self.setAcceptDrops(True)
        self.setStyleSheet("border: 2px solid transparent;")  # Default border (transparent)
        self.update_selection_state()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Toggle selection for the current attribute
            self.main_window.toggle_selection(self.image_path)

            # Start drag event (optional, if you still want drag-and-drop)
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setText(self.image_path)
            drag.setMimeData(mime_data)
            drag.exec(QtCore.Qt.DropAction.MoveAction)

    def update_selection_state(self):
        # Update the border based on whether the image is selected for the current attribute
        if self.image_path in self.main_window.selected_images.get(self.main_window.current_attribute, set()):
            self.setStyleSheet("border: 2px solid red;")  # Red border for selected images
        else:
            self.setStyleSheet("border: 2px solid transparent;")  # Transparent border for unselected images


class ThresholdDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Threshold")
        self.setModal(True)
        self.threshold = 0.5  # Default threshold

        layout = QtWidgets.QVBoxLayout(self)

        # Threshold input
        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.1)
        self.threshold_input.setValue(self.threshold)
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


class RoofClusteringApp(QtWidgets.QMainWindow):
    
    def __init__(self, features_file, root_folder, schema_file, max_samples=500, n_images_per_row=8, image_height=150, image_width=150, window_height=900, window_width=1400):
        super().__init__()
        self.features_file = features_file
        self.root_folder = root_folder
        self.schema_file = schema_file
        self.max_samples = max_samples  # Limit number of images to avoid slowdown
        self.n_images_per_row = n_images_per_row  # Number of images per row
        self.image_height = image_height
        self.image_width = image_width
        self.window_height = window_height
        self.window_width = window_width
        self.load_schema()
        self.current_attribute = list(self.schema.keys())[0]  # Default to first attribute
        self.current_cluster = "undefined"
        self.selected_images = {}  # Dictionary to store selected images per attribute
        
        # Load or compute features
        self.load_or_compute_features()
        
        # Assign images to clusters
        self.assign_images_to_clusters()
        
        # Initialize UI
        self.init_ui()


    def save(self):
        # Save updated assignments
        torch.save({
            'features': self.features,
            'image_paths': self.image_paths,
            'assignments': self.assignments,
            'selected_images': self.selected_images
        }, self.features_file)

    def load_schema(self):
        with open(self.schema_file, 'r') as f:
            self.schema = json.load(f)
        self.schema["undefined"] = ["undefined"]
    
    def load_or_compute_features(self):
        if os.path.exists(self.features_file):
            data = torch.load(self.features_file)
            self.features = data['features'][:self.max_samples]
            self.image_paths = data['image_paths'][:self.max_samples]
            self.assignments = data.get('assignments', {})
            self.selected_images = data.get('selected_images', {})  # Load selected images per attribute
        else:
            images_folder = self.root_folder
            self.image_paths = self.get_all_image_paths(images_folder)[:self.max_samples]
            self.features = self.extract_features(self.image_paths)
            self.assignments = {}
            self.selected_images = {}  # Initialize empty dictionary for selected images
            # save
            self.save()
    
    def get_all_image_paths(self, folder):
        image_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def assign_images_to_clusters(self):
        self.clusters = {key: {val: [] for val in values + ["undefined"]} for key, values in self.schema.items()}
        for i, img_path in enumerate(self.image_paths):
            assigned_value = self.assignments.get(img_path, {}).get(self.current_attribute, "undefined")
            self.clusters[self.current_attribute][assigned_value].append(i)

    def init_ui(self):
        self.setWindowTitle("Roof Clustering by Attribute")
        self.setGeometry(100, 100, self.window_width, self.window_height)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Dropdown for attribute selection
        self.attribute_selector = QtWidgets.QComboBox()
        self.attribute_selector.addItems(self.schema.keys())
        self.attribute_selector.currentTextChanged.connect(self.change_attribute)
        layout.addWidget(self.attribute_selector)
        
        # Auto Classify group box
        auto_classify_group = QtWidgets.QGroupBox("Auto Classify")
        auto_classify_layout = QtWidgets.QHBoxLayout()
        
        # OOD button
        self.ood_button = QtWidgets.QPushButton("OOD")
        self.ood_button.setFixedHeight(50)
        self.ood_button.clicked.connect(self.run_ood_classification)
        auto_classify_layout.addWidget(self.ood_button)
        
        # Random Forest button
        self.rf_button = QtWidgets.QPushButton("RandomForest")
        self.rf_button.setFixedHeight(50)
        self.rf_button.clicked.connect(self.run_random_forest_classification)
        auto_classify_layout.addWidget(self.rf_button)
        
        auto_classify_group.setLayout(auto_classify_layout)
        layout.addWidget(auto_classify_group)
        
        # Cluster selection buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.cluster_buttons = {}
        self.update_cluster_buttons()
        layout.addLayout(self.button_layout)
        
        # Image display area
        self.image_scroll_area = QtWidgets.QScrollArea()
        self.image_container = QtWidgets.QWidget()
        self.image_container.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.image_layout = QtWidgets.QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)
        self.image_scroll_area.setWidgetResizable(True)
        layout.addWidget(self.image_scroll_area)
        
        self.display_cluster_images()

    def change_attribute(self, attribute):
        self.current_attribute = attribute
        self.current_cluster = self.schema[self.current_attribute][0]  # Reset to the first cluster value
        self.assign_images_to_clusters()
        self.update_cluster_buttons()  # Update the cluster buttons
        self.display_cluster_images()  # Display images for the new attribute and cluster

    def run_ood_classification(self):
        # Open threshold dialog
        dialog = ThresholdDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            threshold = dialog.get_threshold()
            
            # Get selected feature vectors for the current cluster
            selected_indices = self.clusters[self.current_attribute].get(self.current_cluster, [])
            selected_features = self.features[selected_indices]
            
            # Get unselected feature vectors from other clusters and not verified samples from the current cluster
            unselected_indices = []
            for cluster, indices in self.clusters[self.current_attribute].items():
                for idx in indices:
                    image_path = self.image_paths[idx]
                    if image_path not in self.selected_images.get(self.current_attribute, set()):
                        unselected_indices.append(idx)
            unselected_features = self.features[unselected_indices]
            
            # Use OOD4Inclusion to classify unselected samples
            ood_classifier = OOD4Inclusion()
            ood_classifier.set_clean_distribution(self.current_cluster, selected_features)
            inlier_mask, _ = ood_classifier.evaluate_new_samples(self.current_cluster, unselected_features, threshold)
            
            # Assign inliers to the current cluster and outliers to "undefined"
            for idx, is_inlier in zip(unselected_indices, inlier_mask):
                image_path = self.image_paths[idx]
                if image_path not in self.assignments:
                    self.assignments[image_path] = {}  # Initialize if not present
                if is_inlier:
                    self.assignments[image_path][self.current_attribute] = self.current_cluster
                else:
                    self.assignments[image_path][self.current_attribute] = "undefined"
            
            # Save updated assignments
            self.save()
            
            # Refresh the UI
            self.assign_images_to_clusters()
            self.display_cluster_images()

    def run_random_forest_classification(self):
        # Open Random Forest parameters dialog
        dialog = RandomForestDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            # Prepare training data (verified samples)
            X_train = []
            y_train = []
            for cluster, indices in self.clusters[self.current_attribute].items():
                for idx in indices:
                    image_path = self.image_paths[idx]
                    if image_path in self.selected_images.get(self.current_attribute, set()):
                        X_train.append(self.features[idx].numpy())
                        y_train.append(cluster)
            
            if not X_train:
                QtWidgets.QMessageBox.warning(self, "Error", "No verified samples found for training.")
                return
            
            # Train Random Forest classifier
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            clf.fit(X_train, y_train)
            
            # Classify non-selected samples
            non_selected_indices = []
            for cluster, indices in self.clusters[self.current_attribute].items():
                for idx in indices:
                    image_path = self.image_paths[idx]
                    if image_path not in self.selected_images.get(self.current_attribute, set()):
                        non_selected_indices.append(idx)
            non_selected_features = self.features[non_selected_indices]
            
            if non_selected_features.shape[0] > 0:
                predictions = clf.predict(non_selected_features.numpy())
                
                # Assign predictions to non-selected samples
                for idx, pred in zip(non_selected_indices, predictions):
                    image_path = self.image_paths[idx]
                    if image_path not in self.assignments:
                        self.assignments[image_path] = {}  # Initialize if not present
                    self.assignments[image_path][self.current_attribute] = pred
            
            # Save updated assignments
            self.save()
            
            # Refresh the UI
            self.assign_images_to_clusters()
            self.display_cluster_images()

    def update_cluster_buttons(self):
        # Clear existing buttons
        for i in reversed(range(self.button_layout.count())):
            self.button_layout.itemAt(i).widget().deleteLater()
        
        # Add new buttons for the current attribute's values
        self.cluster_buttons = {}
        for value in self.schema[self.current_attribute] + ["undefined"]:
            button = QtWidgets.QPushButton(value)
            button.setFixedHeight(50)
            button.clicked.connect(partial(self.change_cluster, value))
            button.setAcceptDrops(True)  # Enable drop events
            button.dragEnterEvent = self.drag_enter_event
            button.dropEvent = partial(self.drop_event, value)
            self.button_layout.addWidget(button)
            self.cluster_buttons[value] = button
        
        # Highlight the current cluster button
        self.highlight_current_cluster_button()

    def drag_enter_event(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def drop_event(self, target_cluster, event):
        image_path = event.mimeData().text()
        if image_path:
            self.reassign_image_to_cluster(image_path, target_cluster)
            event.acceptProposedAction()

    def reassign_image_to_cluster(self, image_path, target_cluster):
        # Update the assignments dictionary
        if image_path not in self.assignments:
            self.assignments[image_path] = {}  # Initialize if not present
        self.assignments[image_path][self.current_attribute] = target_cluster
        
        # Reassign the image to the new cluster
        self.assign_images_to_clusters()
        
        # Automatically mark the moved image as confirmed for the current attribute
        self.toggle_selection(image_path, force_select=True)
        
        # Save the updated assignments and selected images
        self.save()
        
        # Refresh the UI
        self.display_cluster_images()

    def toggle_selection(self, image_path, force_select=False):
        # Toggle selection for the current attribute
        if self.current_attribute not in self.selected_images:
            self.selected_images[self.current_attribute] = set()
        
        if force_select or image_path not in self.selected_images[self.current_attribute]:
            self.selected_images[self.current_attribute].add(image_path)  # Select
        else:
            self.selected_images[self.current_attribute].discard(image_path)  # Deselect
        
        # Refresh the UI to update the border
        self.display_cluster_images()

    def highlight_current_cluster_button(self):
        # Reset all buttons to default color
        for button in self.cluster_buttons.values():
            button.setStyleSheet("")  # Reset to default style
        
        # Highlight the current cluster button
        if self.current_cluster in self.cluster_buttons:
            self.cluster_buttons[self.current_cluster].setStyleSheet("background-color: green; color: white;")

    def display_cluster_images(self):
        # Clear previous images
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().deleteLater()
        
        if self.current_attribute not in self.clusters:
            return
        
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])[:self.max_samples]
        for idx, image_idx in enumerate(cluster_images):
            image_path = self.image_paths[image_idx]
            pixmap = QtGui.QPixmap(image_path).scaled(self.image_width, self.image_height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label = DraggableLabel(image_path, self)  # Pass the main window reference
            label.setPixmap(pixmap)
            label.update_selection_state()  # Update the border based on selection
            row = idx // self.n_images_per_row
            col = idx % self.n_images_per_row
            self.image_layout.addWidget(label, row, col)

    def change_cluster(self, cluster):
        self.current_cluster = cluster
        self.highlight_current_cluster_button()  # Highlight the new cluster button
        self.display_cluster_images()  # Display images for the new cluster

    def extract_features(self, image_paths):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        model.to('cuda')

        features = []
        for image_path in tqdm(image_paths[:self.max_samples]):
            image = np.array(Image.open(image_path))[:, :, :3]
            img_t = self.transform_image(image)
            img_t = img_t.to('cuda')
            img_t = F.interpolate(img_t, size=(252, 252), mode='bilinear', align_corners=False)  # Resize to multiple of 14
            with torch.no_grad():
                feature = model(img_t).cpu()
            features.append(feature)
        return torch.cat(features, dim=0)
    
    def transform_image(self, image, sz=256):
        trans = A.Compose([A.LongestMaxSize(max_size=sz), A.PadIfNeeded(sz, sz)])
        image_t = trans(image=image)['image']
        image_t = rearrange(image_t, 'h w c -> 1 c h w')
        return torch.tensor(image_t, dtype=torch.float32)


def main():
    app = QtWidgets.QApplication(sys.argv)
    features_file = 'features.pt'
    root_folder = 'segmentation_dataset/cropped_images'
    schema_file = 'schema.json'
    main_window = RoofClusteringApp(features_file, root_folder, schema_file)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()