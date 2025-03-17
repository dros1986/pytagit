import sys
import copy
import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6 import QtCore
from PyQt6.QtWidgets import QToolTip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# from sklearn.manifold import TSNE
from openTSNE import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from functools import partial
import seaborn as sns

sns.set_theme()


class TSNEDialog(QtWidgets.QDialog):
    def __init__(self, feature_vectors, image_paths, selected_images, current_cluster, assignments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive t-SNE Visualization")
        self.setModal(True)

        # Parameters
        self.feature_vectors = feature_vectors
        self.image_paths = image_paths
        self.selected_images = selected_images
        self.current_cluster = current_cluster
        # self.assignments = assignments
        # self.assignments = copy.deepcopy(assignments)
        self.assignments = {}
        self.parent = parent

        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # t-SNE Canvas
        self.figure = Figure(figsize=(15, 13))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Boundaries")
        self.save_button.clicked.connect(self.save_boundaries)
        self.reset_button = QtWidgets.QPushButton("Reset Boundaries")
        self.reset_button.clicked.connect(self.reset_boundaries)
        self.save_button.setFixedHeight(50)
        self.reset_button.setFixedHeight(50)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)

        # Initialize t-SNE plot
        self.ax = self.figure.add_subplot(111)
        self.scatter = None
        self.polygon = None
        self.vertices = []
        self.compute_tsne()

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def compute_tsne(self):
        """Compute t-SNE embedding and plot points."""
        # hide super window
        if self.parent is not None:
            self.parent.setVisible(False)
            print('-'*50)
            print('Computing t-SNE...')
        # radnomly select 1000 indices
        indices = np.random.choice(len(self.feature_vectors), min(2000, len(self.feature_vectors)), replace=False)
        train_data = self.feature_vectors[indices]
        # Compute t-SNE with openTSNE to use less samples in training
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.feature_vectors)-1))
        out = tsne.fit(train_data)
        # transform all data
        self.tsne_embeddings = out.transform(self.feature_vectors)
        # tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.feature_vectors)-1))
        # self.tsne_embeddings = tsne.fit_transform(self.feature_vectors)

        # show super window
        if self.parent is not None:
            self.parent.setVisible(True)

        # Separate points by selection status
        selected_points = []
        unselected_points = []

        for i, path in enumerate(self.image_paths):
            if path in self.selected_images.get(self.current_cluster, set()):
                selected_points.append(self.tsne_embeddings[i])
            else:
                unselected_points.append(self.tsne_embeddings[i])

        selected_points = np.array(selected_points)
        unselected_points = np.array(unselected_points)

        # Plot points
        self.ax.clear()

        if len(unselected_points) > 0:
            self.ax.scatter(
                unselected_points[:, 0],
                unselected_points[:, 1],
                c="blue",
                marker="o",
                label="Unselected",
                alpha=0.5,
            )

        if len(selected_points) > 0:
            self.ax.scatter(
                selected_points[:, 0],
                selected_points[:, 1],
                c="lawngreen",
                marker="x",
                label="Selected",
                linewidths=3,
                s=100,
            )

        self.ax.legend()
        self.canvas.draw()


    def on_hover(self, event):
        """Show image preview on hover."""
        if event.inaxes != self.ax:
            return

        # Find nearest point in t-SNE space
        distances = np.linalg.norm(self.tsne_embeddings - np.array([event.xdata, event.ydata]), axis=1)
        nearest_idx = np.argmin(distances)
        image_path = self.image_paths[nearest_idx]

        # Display image preview
        pixmap = QtGui.QPixmap(image_path).scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"<img src='{image_path}' width='100' height='100'>")

    def on_click(self, event):
        """Handle mouse clicks for drawing boundaries."""
        if event.inaxes != self.ax:
            return

        # Add vertex to polygon
        self.vertices.append((event.xdata, event.ydata))
        if len(self.vertices) > 1:
            if self.polygon:
                self.polygon.remove()
            self.polygon = Polygon(self.vertices, closed=False, edgecolor="red", fill=False)
            self.ax.add_patch(self.polygon)
            self.canvas.draw()

    def save_boundaries(self):
        """Save classification results based on drawn boundaries."""
        if len(self.vertices) < 3:
            QtWidgets.QMessageBox.warning(self, "Error", "Please draw a valid polygon.")
            return

        # Convert vertices to a polygon
        polygon = Polygon(self.vertices, closed=True)

        # Classify points inside the polygon
        for i, point in enumerate(self.tsne_embeddings):
            if polygon.contains_point(point):
                self.assignments[self.image_paths[i]] = self.current_cluster

        # Refresh the UI
        # self.parent.display_cluster_images()
        # close the dialog
        self.close()


    def reset_boundaries(self):
        """Reset drawn boundaries."""
        self.vertices = []
        if self.polygon:
            self.polygon.remove()
            self.polygon = None
        self.canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Example data
    feature_vectors = np.random.rand(100, 512)  # Random feature vectors
    image_paths = [f"image_{i}.jpg" for i in range(100)]
    selected_images = {"cluster_1": set(["image_1.jpg", "image_2.jpg"])}
    current_cluster = "cluster_1"
    assignments = {}

    # dialog = TSNEDialog(feature_vectors, image_paths, selected_images, current_cluster, assignments)
    dialog = TSNEDialog(
        feature_vectors=feature_vectors,
        image_paths=image_paths,
        selected_images=selected_images,
        current_cluster=current_cluster,
        assignments=assignments,
        # parent=self  # Pass the main window as the parent
    )
    dialog.exec()

    sys.exit(app.exec())