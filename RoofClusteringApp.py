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
from Helpers import ThresholdDialog, RFClassifier, CNNClassifier, kNNClassifier


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

        # Main layout to hold both group boxes
        main_layout = QtWidgets.QHBoxLayout()

        # Auto Classify group box
        auto_classify_group = QtWidgets.QGroupBox("Auto Classify")
        auto_classify_group.setMaximumHeight(120)
        auto_classify_layout = QtWidgets.QHBoxLayout()

        self.ood_button = QtWidgets.QPushButton("OOD")
        self.ood_button.setFixedHeight(50)
        self.ood_button.clicked.connect(self.run_ood_classification)
        auto_classify_layout.addWidget(self.ood_button)

        self.classifiers = [RFClassifier(), CNNClassifier(), kNNClassifier()]
        for cur_classifier in self.classifiers:
            cur_button = QtWidgets.QPushButton(cur_classifier.get_name())
            cur_button.setFixedHeight(50)
            partial_fun = partial(self.run_classification, trainer=cur_classifier)
            cur_button.clicked.connect(partial_fun)
            auto_classify_layout.addWidget(cur_button)

        auto_classify_group.setLayout(auto_classify_layout)
        main_layout.addWidget(auto_classify_group, 8)


        # Sampling group box
        sampling_group = QtWidgets.QGroupBox("Resampling")
        sampling_group.setToolTip('With this box it is possible to address imbalanced problems.')
        sampling_group.setMaximumHeight(120)
        sampling_layout = QtWidgets.QVBoxLayout()

        self.sampling_button_group = QtWidgets.QButtonGroup()
        
        self.sampling_none = QtWidgets.QRadioButton("None")
        self.sampling_none.setToolTip('Selecting this option, the dataset will not be balanced and ramain the same.')
        self.sampling_none.setChecked(True)  # Default selection
        self.sampling_over = QtWidgets.QRadioButton("Over")
        self.sampling_over.setToolTip('The system will duplicate samples from minor classes to match major cardinality.')
        self.sampling_under = QtWidgets.QRadioButton("Under")
        self.sampling_under.setToolTip('The system will delete samples from major classes to match minor cardinality.')

        self.sampling_button_group.addButton(self.sampling_none)
        self.sampling_button_group.addButton(self.sampling_over)
        self.sampling_button_group.addButton(self.sampling_under)

        # Reduce spacing between radio buttons
        sampling_layout.setSpacing(1)  # Decrease spacing (default is usually around 6-10)
        sampling_layout.setContentsMargins(10, 2, 2, 2)  # Reduce margins if needed

        sampling_layout.addWidget(self.sampling_none)
        sampling_layout.addWidget(self.sampling_over)
        sampling_layout.addWidget(self.sampling_under)

        sampling_group.setLayout(sampling_layout)
        main_layout.addWidget(sampling_group, 1)

        # Pseudo-labelling group box
        pseudolabelling_group = QtWidgets.QGroupBox("Pseudo-Labelling")
        pseudolabelling_group.setMaximumHeight(120)
        pseudolabelling_layout = QtWidgets.QHBoxLayout()

        self.pseudo_checkbox = QtWidgets.QCheckBox("Pseudo-labelling")
        self.acceptance_threshold_input = QtWidgets.QDoubleSpinBox()
        self.acceptance_threshold_input.setRange(0.0, 1.0)
        self.acceptance_threshold_input.setSingleStep(0.10)
        self.acceptance_threshold_input.setValue(0.7)
        self.acceptance_threshold_input.setEnabled(False)
        self.acceptance_threshold_input.setPrefix("Threshold: ")

        self.num_iterations_input = QtWidgets.QSpinBox()
        self.num_iterations_input.setRange(1, 10000)
        self.num_iterations_input.setSingleStep(20)
        self.num_iterations_input.setValue(20)
        self.num_iterations_input.setEnabled(False)
        self.num_iterations_input.setPrefix("Iterations: ")

        def toggle_pseudo_options(state):
            enabled = state == QtCore.Qt.CheckState.Checked.value
            self.acceptance_threshold_input.setEnabled(enabled)
            self.num_iterations_input.setEnabled(enabled)

        self.pseudo_checkbox.stateChanged.connect(toggle_pseudo_options)

        pseudolabelling_layout.addWidget(self.pseudo_checkbox)
        pseudolabelling_layout.addWidget(self.acceptance_threshold_input)
        pseudolabelling_layout.addWidget(self.num_iterations_input)

        pseudolabelling_group.setLayout(pseudolabelling_layout)
        main_layout.addWidget(pseudolabelling_group, 2)

        layout.addLayout(main_layout)

        # Cluster selection buttons
        self.button_layout = QtWidgets.QHBoxLayout()
        self.cluster_buttons = {}
        self.update_cluster_buttons()
        layout.addLayout(self.button_layout)

        # Page navigation and image display area
        self.page_label = QtWidgets.QLabel(f"Page {self.current_page + 1} / {self.total_pages}")
        self.page_label.setMaximumHeight(50)
        layout.addWidget(self.page_label)

        self.image_container = QtWidgets.QWidget()
        self.image_container.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.image_layout = QtWidgets.QGridLayout(self.image_container)
        layout.addWidget(self.image_container)

        # add selection buttons (Select All, Deselect All, Remove Unselected)
        selection_layout = QtWidgets.QHBoxLayout()

        self.select_all_button = QtWidgets.QPushButton("Select All")
        self.select_all_button.setFixedHeight(50)
        select_fun = partial(self.select_or_deselect_all_images_in_current_page, select=True)
        self.select_all_button.clicked.connect(select_fun)
        selection_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_button.setFixedHeight(50)
        deselect_fun = partial(self.select_or_deselect_all_images_in_current_page, select=False)
        self.deselect_all_button.clicked.connect(deselect_fun)
        selection_layout.addWidget(self.deselect_all_button)

        self.remove_unselected_button = QtWidgets.QPushButton("Remove unselected from \n current page")
        self.remove_unselected_button.setFixedHeight(50)
        self.remove_unselected_button.clicked.connect(self.remove_unselected_images)
        selection_layout.addWidget(self.remove_unselected_button)

        self.remove_unselected_button = QtWidgets.QPushButton("Remove all \n unselected samples")
        self.remove_unselected_button.setFixedHeight(50)
        self.remove_unselected_button.clicked.connect(self.remove_all_unselected_images)
        selection_layout.addWidget(self.remove_unselected_button)

        # Add save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setFixedHeight(50)
        save_fn = partial(self.save, is_button=True)
        self.save_button.clicked.connect(save_fn)
        selection_layout.addWidget(self.save_button)
        # layout.addWidget(self.save_button)

        # Add the selection layout just before the save button
        layout.addLayout(selection_layout)

        # Connect keyboard and mouse events for navigation
        self.shortcut_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcut_left.activated.connect(self.navigate_previous_page)
        self.shortcut_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcut_right.activated.connect(self.navigate_next_page)
        self.shortcut_select = QtGui.QShortcut(QtGui.QKeySequence(" "), self)
        self.shortcut_select.activated.connect(self.select_or_deselect_images_in_page_on_keypress)
        self.image_container.wheelEvent = self.handle_mouse_wheel

        self.display_cluster_images()

    @property
    def total_pages(self):
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])
        return (len(cluster_images) + self.page_size - 1) // self.page_size
    
    @property
    def num_selected_images_in_page(self):
        # Calculate start and end indices for the current page
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])[:self.max_samples]
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(cluster_images))
        # if no images selected for current attribute, return 0
        if self.current_attribute not in self.selected_images: 
            return 0
        # count selected images
        num_selected = 0
        for cur_image_idx in cluster_images[start_idx:end_idx]:
            if self.image_paths[cur_image_idx] in self.selected_images[self.current_attribute]:
                num_selected += 1
        # return
        return num_selected
    

    def get_visible_images(self, return_idx=False):
        # Calculate start and end indices for the current page
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])[:self.max_samples]
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(cluster_images))
        # get filenames
        filenames = [self.image_paths[image_idx] for image_idx in cluster_images[start_idx:end_idx]]
        # return
        if return_idx:
            return filenames, list(range(start_idx,end_idx))
        return filenames
    

    def select_or_deselect_all_images_in_current_page(self, select=True):
        # get visible images
        visible_images = self.get_visible_images()
        # select all images in the current page
        for image_path in visible_images:
            # create set of selected images if it does not exist
            if self.current_attribute not in self.selected_images:
                self.selected_images[self.current_attribute] = set()
            # add current image to selected images
            if select:
                self.selected_images[self.current_attribute].add(image_path)
            else:
                self.selected_images[self.current_attribute].discard(image_path)
        # display cluster images
        self.display_cluster_images()


    def select_or_deselect_images_in_page_on_keypress(self):
        if self.num_selected_images_in_page > 0:
            self.select_or_deselect_all_images_in_current_page(select=False)
        else:
            self.select_or_deselect_all_images_in_current_page(select=True)



    def remove_unselected_images(self):
        # initialize selected images if not already done
        if self.current_attribute not in self.selected_images:
            self.selected_images[self.current_attribute] = set()
        # get visible images
        visible_images = self.get_visible_images()
        # for each image
        for cur_image in visible_images:
            # if it is not selected
            if cur_image not in self.selected_images[self.current_attribute]:
                # move it in the undefined cluster
                self.reassign_image_to_cluster(cur_image, 'undefined', mark_as_confirmed=False)
        # move to last page
        if self.current_page >= self.total_pages:
            self.current_page = max(0,self.total_pages-1)
            self.display_cluster_images()

    
    def remove_all_unselected_images(self):
        # initialize selected images if not already done
        if self.current_attribute not in self.selected_images:
            self.selected_images[self.current_attribute] = set()
        # get visible images
        cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])[:self.max_samples]
        filenames = [self.image_paths[image_idx] for image_idx in cluster_images]
        # for each image
        for cur_image in filenames:
            # if it is not selected
            if cur_image not in self.selected_images[self.current_attribute]:
                # move it in the undefined cluster
                self.reassign_image_to_cluster(cur_image, 'undefined', mark_as_confirmed=False)
        # move to last page
        if self.current_page >= self.total_pages:
            self.current_page = max(0,self.total_pages-1)
            self.display_cluster_images()



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

        # Collect all verified samples, excluding "undefined" examples
        for cluster, indices in self.clusters[self.current_attribute].items():
            if cluster == 'undefined':  # Skip undefined examples
                continue
            for idx in indices:
                image_path = self.image_paths[idx]
                if image_path in self.selected_images.get(self.current_attribute, set()):
                    X_train.append(self.features[idx].numpy())
                    y_train.append(cluster)
                    filenames.append(image_path)

        # If no samples are available, return None
        if not X_train:
            QtWidgets.QMessageBox.warning(self, "Error", "No verified samples found for training.")
            return None

        # Convert to numpy arrays
        X_train = np.vstack(X_train)
        le = preprocessing.LabelEncoder()
        unique_class_names = list(set(y_train))
        le.fit(unique_class_names)
        y_train_encoded = le.transform(y_train)

        # Count the number of samples per class
        class_counts = {cls: sum(y_train_encoded == cls_id) for cls_id, cls in enumerate(le.classes_)}

        # Determine the sampling strategy
        if self.sampling_none.isChecked():
            # No resampling
            pass
        elif self.sampling_over.isChecked():
            # Over-sampling: Duplicate samples from minor classes
            max_count = max(class_counts.values())
            for cls_id, cls in enumerate(le.classes_):
                current_count = class_counts[cls]
                if current_count < max_count:
                    # Calculate how many additional samples are needed
                    num_to_add = max_count - current_count
                    cls_indices = np.where(y_train_encoded == cls_id)[0]
                    if len(cls_indices) == 0:  # Ensure the class has samples
                        print(f"Warning: Class '{cls}' has no samples. Skipping upsampling.")
                        continue
                    additional_samples = np.random.choice(cls_indices, size=num_to_add, replace=True)
                    X_train = np.vstack([X_train, X_train[additional_samples]])
                    y_train_encoded = np.concatenate([y_train_encoded, [cls_id] * num_to_add])
                    filenames.extend([filenames[i] for i in additional_samples])
        elif self.sampling_under.isChecked():
            # Under-sampling: Reduce samples from major classes
            min_count = min(class_counts.values())
            filtered_X_train = []
            filtered_y_train_encoded = []
            filtered_filenames = []
            for cls_id, cls in enumerate(le.classes_):
                cls_indices = np.where(y_train_encoded == cls_id)[0]
                if len(cls_indices) > min_count:
                    # Randomly select a subset of samples
                    selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
                else:
                    selected_indices = cls_indices
                filtered_X_train.append(X_train[selected_indices])
                filtered_y_train_encoded.extend([cls_id] * len(selected_indices))
                filtered_filenames.extend([filenames[i] for i in selected_indices])
            X_train = np.vstack(filtered_X_train)
            y_train_encoded = np.array(filtered_y_train_encoded)
            filenames = filtered_filenames

        # Get number of classes
        num_classes = len(unique_class_names)

        # Add "undefined" class for consistency
        le.classes_ = np.append(le.classes_, "undefined")

        # Find class ID of "undefined"
        id_undefined_class = int(le.transform(['undefined'])[0])

        return X_train, y_train_encoded, filenames, num_classes, id_undefined_class, le
    


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
        # If accepted
        if not dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return
        # get parameters
        params = dialog.get_parameters()

        # Check if pseudolabeling is enabled
        pseudo_enabled = self.pseudo_checkbox.isChecked()
        acceptance_threshold = self.acceptance_threshold_input.value()
        num_iterations = self.num_iterations_input.value()

        # Get training features
        ret_vals = self.get_training_features()
        if ret_vals is None:
            print('No examples selected for this feature.')
            return
        # split returned values
        X_train, y_train, filenames, num_classes, id_undefined_class, le = ret_vals

        # print number of samples for each class
        unique, counts = np.unique(y_train, return_counts=True)
        print('-'*100)
        print('Number of samples for each class:')
        for u,c in zip(unique, counts):
            print(u, '-->', c)

        if X_train is None:  # Check if no samples are available for training
            return

        # Train the classifier
        self.setVisible(False)
        training_performed = trainer.train(params, X_train, y_train, filenames, id_undefined_class, num_classes)
        # self.setVisible(True)
        if not training_performed:
            self.setVisible(True)
            return

        # Pseudolabeling loop if enabled
        if pseudo_enabled:
            print('-'*100)
            non_selected_indices, non_selected_filenames, non_selected_features = self.get_non_selected_features()

            for n_iter, iteration in enumerate(range(num_iterations)):
                # if the number of non_selected_samples is 0, quit
                if len(non_selected_features) == 0: 
                    print('No more untagged samples. Quitting pseudo labeling.')
                    break
                # Classify non-selected samples
                predictions, confidence_scores = trainer.classify(params, non_selected_filenames, non_selected_features, id_undefined_class)

                # Filter predictions based on confidence (threshold)
                confident_predictions = []
                confident_indices = []

                # filter by confidence (TODO: can be optimized)
                for idx, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                    if conf >= acceptance_threshold:
                        confident_predictions.append(pred)
                        # confident_indices.append(non_selected_indices[idx])
                        confident_indices.append(idx)

                # Update assignments with confident predictions
                for idx, pred in zip(confident_indices, confident_predictions):
                    image_path = self.image_paths[non_selected_indices[idx]]
                    if image_path not in self.assignments:
                        self.assignments[image_path] = {}
                    self.assignments[image_path][self.current_attribute] = str(le.inverse_transform([pred])[0])

                # Add newly labeled samples to the training set
                new_X_train = non_selected_features[confident_indices].numpy()
                new_y_train = np.array(confident_predictions)
                new_filenames = [non_selected_filenames[i] for i in confident_indices]

                # Remove labeled samples from the pool of unlabeled samples
                # non_selected_indices = [idx for idx in non_selected_indices if idx not in confident_indices]
                # non_selected_features = non_selected_features[[i for i in range(len(non_selected_features)) if i not in confident_indices]]
                # non_selected_filenames = [fn for fn in non_selected_filenames if fn not in new_filenames]

                # Retrain the model with the updated dataset
                if len(new_X_train) > 0:
                    num_of_new_samples = len(new_X_train)
                    new_X_train = np.vstack([new_X_train, X_train])
                    new_y_train = np.concatenate([new_y_train, y_train])
                    new_filenames.extend(filenames)
                    # check and print
                    assert(id_undefined_class not in new_y_train)
                    print(f'Iteration {n_iter+1}/{num_iterations} of pseudolabeling. Using {len(X_train)} real, {num_of_new_samples} fake, {len(new_X_train)} total samples.')
                    # train again
                    trainer.train(params, new_X_train, new_y_train, new_filenames, id_undefined_class, num_classes)
                else:
                    print(f'Iteration {n_iter+1}/{num_iterations} of pseudolabeling. No more pseudolabels. Stopping.')
                    break

            # check if threshold is inside params
            if 'threshold' in params:
                acceptance_threshold = params['threshold']
            else:
                acceptance_threshold = 0

        # else:
        # at the end, classify
        # Without pseudolabeling, classify all non-selected samples once
        non_selected_indices, non_selected_filenames, non_selected_features = self.get_non_selected_features()
        predictions, confidences = trainer.classify(params, non_selected_filenames, non_selected_features, id_undefined_class)

        predictions[confidences < acceptance_threshold] = id_undefined_class

        # From label IDs to label names
        predictions = [str(v) for v in le.inverse_transform(predictions)]

        # Assign predictions to non-selected samples
        for idx, pred in zip(non_selected_indices, predictions):
            image_path = self.image_paths[idx]
            if image_path not in self.assignments:
                self.assignments[image_path] = {}
            self.assignments[image_path][self.current_attribute] = pred

        # Save updated assignments
        # self.save()

        # show gui
        self.setVisible(True)

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

    
    def reassign_image_to_cluster(self, image_path, target_cluster, mark_as_confirmed=True):
        # Update the assignments dictionary
        if image_path not in self.assignments:
            self.assignments[image_path] = {}
        self.assignments[image_path][self.current_attribute] = target_cluster

        # Reassign the image to the new cluster
        self.assign_images_to_clusters()

        # Automatically mark the moved image as confirmed for the current attribute
        if mark_as_confirmed:
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


    # @property
    # def total_pages(self):
    #     cluster_images = self.clusters[self.current_attribute].get(self.current_cluster, [])
    #     return (len(cluster_images) + self.page_size - 1) // self.page_size
    

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