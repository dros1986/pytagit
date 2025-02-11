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
from CNNTrainer import ImageDataset, transform_fun
from Helpers import ThresholdDialog, RFClassifier, CNNClassifier


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




class RoofClusteringApp(QtWidgets.QMainWindow):
    def __init__(self, features_file, root_folder, schema_file, max_samples=10000, n_images_per_row=8, image_height=150, image_width=150, window_height=900, window_width=1400, n_rows_per_page=5):
        super().__init__()
        self.features_file = features_file
        self.root_folder = root_folder
        self.schema_file = schema_file
        self.max_samples = max_samples
        self.n_images_per_row = n_images_per_row
        self.image_height = image_height
        self.image_width = image_width
        self.window_height = window_height
        self.window_width = window_width
        self.n_rows_per_page = n_rows_per_page
        self.page_size = self.n_rows_per_page * self.n_images_per_row
        self.autosave = False
        self.current_page = 0  # Track the current page
        self.load_schema()
        self.current_attribute = list(self.schema.keys())[0]
        self.current_cluster = "undefined"
        self.selected_images = {}
        self.labels = {}
        self.load_or_compute_features()
        self.assign_images_to_clusters()
        self.init_ui()

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
        self.attribute_selector.setStyleSheet("QComboBox { height: 25px; font-size: 16px; }")
        layout.addWidget(self.attribute_selector)

        # Auto Classify group box
        auto_classify_group = QtWidgets.QGroupBox("Auto Classify")
        auto_classify_layout = QtWidgets.QHBoxLayout()
        self.ood_button = QtWidgets.QPushButton("OOD")
        self.ood_button.setFixedHeight(50)
        self.ood_button.clicked.connect(self.run_ood_classification)
        auto_classify_layout.addWidget(self.ood_button)
        self.classifiers = [RFClassifier(), CNNClassifier()]
        for cur_classifier in self.classifiers:
            cur_button = QtWidgets.QPushButton(cur_classifier.get_name())
            cur_button.setFixedHeight(50)
            partial_fun = partial(self.run_classification, trainer=cur_classifier)
            cur_button.clicked.connect(partial_fun)
            auto_classify_layout.addWidget(cur_button)
        auto_classify_group.setLayout(auto_classify_layout)
        layout.addWidget(auto_classify_group)

        # Cluster selection buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.cluster_buttons = {}
        self.update_cluster_buttons()
        layout.addLayout(self.button_layout)

        # Page navigation and image display area
        self.page_label = QtWidgets.QLabel(f"Page {self.current_page + 1} / {self.total_pages}")
        layout.addWidget(self.page_label)

        self.image_container = QtWidgets.QWidget()
        self.image_container.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.image_layout = QtWidgets.QGridLayout(self.image_container)
        layout.addWidget(self.image_container)

        # Add save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setFixedHeight(50)
        save_fn = partial(self.save, is_button=True)
        self.save_button.clicked.connect(save_fn)
        layout.addWidget(self.save_button)

        # Connect keyboard and mouse events for navigation
        self.shortcut_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcut_left.activated.connect(self.navigate_previous_page)
        self.shortcut_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcut_right.activated.connect(self.navigate_next_page)
        self.image_container.wheelEvent = self.handle_mouse_wheel

        self.display_cluster_images()

    @property
    def total_pages(self):
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])
        return (len(cluster_images) + self.page_size - 1) // self.page_size

    def display_cluster_images(self):
        if self.current_attribute not in self.clusters:
            return

        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])[:self.max_samples]

        # Clear existing labels
        for label in list(self.labels.values()):
            self.image_layout.removeWidget(label)
            label.deleteLater()
        self.labels.clear()

        # Calculate start and end indices for the current page
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(cluster_images))

        # Load images for the current page
        for idx, image_idx in enumerate(cluster_images[start_idx:end_idx]):
            image_path = self.image_paths[image_idx]
            pixmap = QtGui.QPixmap(image_path).scaled(self.image_width, self.image_height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label = DraggableLabel(image_path, self)
            label.setPixmap(pixmap)
            label.update_selection_state()
            self.labels[image_path] = label
            row = idx // self.n_images_per_row
            col = idx % self.n_images_per_row
            self.image_layout.addWidget(self.labels[image_path], row, col)

        # Update page label
        self.page_label.setText(f"Page {self.current_page + 1} / {self.total_pages}")

    def navigate_next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_cluster_images()

    def navigate_previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_cluster_images()

    def handle_mouse_wheel(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.navigate_previous_page()
        elif delta < 0:
            self.navigate_next_page()


    def save(self, is_button=False):
        # if not autosaving, save only if button pressed
        if not self.autosave and not is_button:
            return
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
            data = torch.load(self.features_file, weights_only=False)
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
            self.save(is_button=True)
    
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


    def get_training_features(self):    
        # Prepare training data (verified samples)
        X_train = []
        y_train = []
        filenames = []
        for cluster, indices in self.clusters[self.current_attribute].items():
            for idx in indices:
                image_path = self.image_paths[idx]
                if image_path in self.selected_images.get(self.current_attribute, set()):
                    X_train.append(self.features[idx].numpy())
                    y_train.append(cluster)
                    filenames.append(image_path)
        # if no samples, return
        if not X_train:
            QtWidgets.QMessageBox.warning(self, "Error", "No verified samples found for training.")
            return
        # concatenate
        X_train = np.vstack(X_train)
        # encode labels
        le = preprocessing.LabelEncoder()
        unique_class_names = list(set(y_train + ['undefined']))
        le.fit(unique_class_names)
        y_train = le.transform(y_train)
        # get number of classes
        num_classes = len(unique_class_names)
        # find class id of undefined
        id_undefined_class = int(le.transform(['undefined'])[0])
        # return
        return X_train, y_train, filenames, num_classes, id_undefined_class, le
    


    def get_non_selected_features(self):
        # Classify non-selected samples
        non_selected_indices = []
        non_selected_filenames = []
        for cluster, indices in self.clusters[self.current_attribute].items():
            for idx in indices:
                image_path = self.image_paths[idx]
                if image_path not in self.selected_images.get(self.current_attribute, set()):
                    non_selected_indices.append(idx)
                    non_selected_filenames.append(image_path)
        non_selected_features = self.features[non_selected_indices]

        # return all
        return non_selected_indices, non_selected_filenames, non_selected_features



    def run_classification(self, trainer):
        # Open dialog
        dialog = trainer.get_dialog(self)
        # if accepted
        if not dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return
        # get parameters
        params = dialog.get_parameters()
        # get train features
        X_train, y_train, filenames, num_classes, id_undefined_class, le = self.get_training_features()

        # train
        self.setVisible(False)
        training_performed = trainer.train(params, X_train, y_train, filenames, id_undefined_class, num_classes)
        self.setVisible(True)
        if not training_performed: return
        # get non selected features
        non_selected_indices, non_selected_filenames, non_selected_features = self.get_non_selected_features()
        # classify them
        predictions = trainer.classify(params, non_selected_filenames, non_selected_features, id_undefined_class)
        # from label ids to label names
        predictions = [str(v) for v in le.inverse_transform(predictions)]

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
            button.dragEnterEvent = partial(self.drag_enter_event, button)
            button.dragLeaveEvent = partial(self.drag_leave_event, button)
            button.dropEvent = partial(self.drop_event, value, button)
            self.button_layout.addWidget(button)
            self.cluster_buttons[value] = button
        
        # Highlight the current cluster button
        self.highlight_current_cluster_button()

    def drag_enter_event(self, button, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
            button.original_style = button.styleSheet()
            button.setStyleSheet("background-color: yellow;")

    def drag_leave_event(self, button, event):
        button.setStyleSheet(button.original_style)

    def drop_event(self, target_cluster, button, event):
        image_path = event.mimeData().text()
        if image_path:
            self.reassign_image_to_cluster(image_path, target_cluster)
            event.acceptProposedAction()
            button.setStyleSheet(button.original_style)

    
    def reassign_image_to_cluster(self, image_path, target_cluster):
        # Update the assignments dictionary
        if image_path not in self.assignments:
            self.assignments[image_path] = {}
        self.assignments[image_path][self.current_attribute] = target_cluster

        # Reassign the image to the new cluster
        self.assign_images_to_clusters()

        # Automatically mark the moved image as confirmed for the current attribute
        self.toggle_selection(image_path, force_select=True)

        # Save the updated assignments and selected images
        self.save()

        # Incrementally update the UI for the moved image
        self.update_image_position(image_path)


    def update_image_position(self, image_path):
        # Find the current index of the image in the grid layout
        for idx, label in enumerate(self.labels.values()):
            if label.image_path == image_path:
                old_row = idx // self.n_images_per_row
                old_col = idx % self.n_images_per_row
                break
        else:
            return  # Image not found in the current layout

        # Remove the image from its old position
        self.image_layout.removeWidget(self.labels[image_path])
        self.labels[image_path].setParent(None)

        # Find the new index for the image in the current cluster
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])
        if image_path not in cluster_images:
            return  # Image does not belong to the current cluster

        new_idx = cluster_images.index(self.image_paths.index(image_path))
        new_row = new_idx // self.n_images_per_row
        new_col = new_idx % self.n_images_per_row

        # Add the image to its new position
        self.image_layout.addWidget(self.labels[image_path], new_row, new_col)


    def toggle_selection(self, image_path, force_select=False):
        # Toggle selection for the current attribute
        if self.current_attribute not in self.selected_images:
            self.selected_images[self.current_attribute] = set()
        
        if force_select or image_path not in self.selected_images[self.current_attribute]:
            self.selected_images[self.current_attribute].add(image_path)  # Select
        else:
            self.selected_images[self.current_attribute].discard(image_path)  # Deselect
        
        # Refresh the UI to update the border
        if image_path in self.labels:
            self.labels[image_path].update_selection_state()


    def highlight_current_cluster_button(self):
        # Reset all buttons to default color
        for button in self.cluster_buttons.values():
            button.setStyleSheet("")  # Reset to default style
        
        # Highlight the current cluster button
        if self.current_cluster in self.cluster_buttons:
            self.cluster_buttons[self.current_cluster].setStyleSheet("background-color: green; color: white;")


    @property
    def total_pages(self):
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])
        return (len(cluster_images) + self.page_size - 1) // self.page_size


    def change_cluster(self, cluster):
        self.current_cluster = cluster
        self.highlight_current_cluster_button()  # Highlight the new cluster button
        # Reset current_page if it's out of bounds for the new cluster
        total_pages = self.total_pages
        if self.current_page >= total_pages:
            self.current_page = 0  # Reset to the first page
        self.display_cluster_images()  # Display images for the new cluster

    def extract_features(self, image_paths, batch_size=42, device='cuda'):
        # define model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        model.to(device)
        # define transform
        transform = partial(transform_fun, train=False, sz=252)
        # define dataset
        dataset = ImageDataset(image_paths[:self.max_samples], labels=None, transform=transform)
        # define dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # init features
        features = []
        # transform each batch
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                feature = model(images).cpu()
            features.append(feature)
        # concat and return
        return torch.cat(features, dim=0)


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