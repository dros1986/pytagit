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
from CNNTrainer import train_cnn, classify_with_cnn  # Import the CNN training function


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
        # if self.image_path in self.main_window.selected_images.get(self.main_window.current_attribute, set()):
        if self.image_path in self.main_window.selected_images:
            self.setStyleSheet("border: 2px solid red;")  # Red border for selected images
        else:
            self.setStyleSheet("border: 2px solid transparent;")  # Transparent border for unselected images





class ImageGallery(QtWidgets.QMainWindow):
    
    def __init__(self, root_folder):
        super().__init__()
        self.root_folder = root_folder
        self.max_samples = 1000
        self.n_images_per_row = 8
        self.image_height = 150
        self.image_width = 150
        self.window_height = 900
        self.window_width = 1400

        # get filenames
        self.image_paths = self.get_all_image_paths(self.root_folder)[:self.max_samples]
        self.assignments = {}
        self.selected_images = set()
        
        # Initialize UI
        self.init_ui()


    
    def get_all_image_paths(self, folder):
        image_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    


    def init_ui(self):
        self.setWindowTitle("Image Gallery")
        self.setGeometry(100, 100, self.window_width, self.window_height)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
    
        # Image display area
        self.image_scroll_area = QtWidgets.QScrollArea()
        self.image_container = QtWidgets.QWidget()
        self.image_container.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        self.image_layout = QtWidgets.QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)
        self.image_scroll_area.setWidgetResizable(True)
        layout.addWidget(self.image_scroll_area)

        # view images
        self.display_images()



    def toggle_selection(self, image_path, force_select=False):
        
        if force_select or image_path not in self.selected_images:
            self.selected_images.add(image_path)  # Select
        else:
            self.selected_images.discard(image_path)  # Deselect
        
        # Refresh the UI to update the border
        self.display_images()



    def display_images(self):
        # Clear previous images
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().deleteLater()

        # view each image
        for idx, image_path in enumerate(self.image_paths):
            pixmap = QtGui.QPixmap(image_path).scaled(self.image_width, self.image_height, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            label = DraggableLabel(image_path, self)  # Pass the main window reference
            label.setPixmap(pixmap)
            label.update_selection_state()  # Update the border based on selection
            row = idx // self.n_images_per_row
            col = idx % self.n_images_per_row
            self.image_layout.addWidget(label, row, col)


def main():
    app = QtWidgets.QApplication(sys.argv)
    root_folder = 'segmentation_dataset/cropped_images'
    main_window = ImageGallery(root_folder)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()