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


class DraggableLabel(QtWidgets.QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setText(self.image_path)
            drag.setMimeData(mime_data)
            drag.exec(QtCore.Qt.DropAction.MoveAction)


class RoofClusteringApp(QtWidgets.QMainWindow):
    
    def __init__(self, features_file, root_folder, schema_file):
        super().__init__()
        self.features_file = features_file
        self.root_folder = root_folder
        self.schema_file = schema_file
        self.max_samples = 300  # Limit number of images to avoid slowdown
        self.n_images_per_row = 10  # Number of images per row
        self.load_schema()
        self.current_attribute = list(self.schema.keys())[0]  # Default to first attribute
        self.current_cluster = "undefined"
        self.selected_images = set()
        
        # Load or compute features
        self.load_or_compute_features()
        
        # Assign images to clusters
        self.assign_images_to_clusters()
        
        # Initialize UI
        self.init_ui()

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
        else:
            images_folder = os.path.join(self.root_folder, 'images')
            self.image_paths = self.get_all_image_paths(images_folder)[:self.max_samples]
            self.features = self.extract_features(self.image_paths)
            self.assignments = {}
            torch.save({'features': self.features, 'image_paths': self.image_paths, 'assignments': self.assignments}, self.features_file)
    
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
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Dropdown for attribute selection
        self.attribute_selector = QtWidgets.QComboBox()
        self.attribute_selector.addItems(self.schema.keys())
        self.attribute_selector.currentTextChanged.connect(self.change_attribute)
        layout.addWidget(self.attribute_selector)
        
        # Cluster selection buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.cluster_buttons = {}
        self.update_cluster_buttons()
        layout.addLayout(self.button_layout)
        
        # Image display area
        self.image_scroll_area = QtWidgets.QScrollArea()
        self.image_container = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)
        self.image_scroll_area.setWidgetResizable(True)
        layout.addWidget(self.image_scroll_area)
        
        self.display_cluster_images()

    def update_cluster_buttons(self):
        # Clear existing buttons
        for i in reversed(range(self.button_layout.count())):
            self.button_layout.itemAt(i).widget().deleteLater()
        
        # Add new buttons for the current attribute's values
        self.cluster_buttons = {}
        for value in self.schema[self.current_attribute] + ["undefined"]:
            button = QtWidgets.QPushButton(value)
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
            self.assignments[image_path] = {}
        self.assignments[image_path][self.current_attribute] = target_cluster
        
        # Reassign the image to the new cluster
        self.assign_images_to_clusters()
        
        # Save the updated assignments
        torch.save({'features': self.features, 'image_paths': self.image_paths, 'assignments': self.assignments}, self.features_file)
        
        # Refresh the UI
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
            pixmap = QtGui.QPixmap(image_path).scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label = DraggableLabel(image_path)
            label.setPixmap(pixmap)
            row = idx // self.n_images_per_row
            col = idx % self.n_images_per_row
            self.image_layout.addWidget(label, row, col)

    def change_attribute(self, attribute):
        self.current_attribute = attribute
        self.current_cluster = self.schema[self.current_attribute][0]  # Reset to the first cluster value
        self.assign_images_to_clusters()
        self.update_cluster_buttons()  # Update the cluster buttons
        self.display_cluster_images()  # Display images for the new attribute and cluster

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
    root_folder = 'segmentation_dataset'
    schema_file = 'schema.json'
    main_window = RoofClusteringApp(features_file, root_folder, schema_file)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()