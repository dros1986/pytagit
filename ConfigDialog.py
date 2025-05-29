import sys
import json
import os
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor


CONFIG_HISTORY_PATH = os.path.expanduser("~/.tagit_config_history.json")
CONFIG_TEMPLATE = {
    "class": ["Airport", "BareLand", "BaseballField"],
    "image_clarity": ["bad", "average", "good"]
}
MAX_HISTORY = 10

class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, features_file="", root_folder="", schema_file=None):
        super().__init__()
        self.setWindowTitle("TAGIT Configuration")
        self.setMinimumWidth(800)
        self.layout = QtWidgets.QVBoxLayout(self)

        # Features file
        self.features_input = QtWidgets.QLineEdit(features_file)
        browse_features = QtWidgets.QPushButton("Browse")
        self.features_input.setPlaceholderText("Path to file where features will be saved/loaded (e.g., dataset/features.pt)")
        browse_features.clicked.connect(self.browse_features)
        self.add_labeled_row("Features File:", self.features_input, browse_features)

        # Root folder
        self.root_input = QtWidgets.QLineEdit(root_folder)
        browse_root = QtWidgets.QPushButton("Browse")
        self.root_input.setPlaceholderText("Path to the root folder with images (e.g., dataset/images/)")
        browse_root.clicked.connect(self.browse_root)
        self.add_labeled_row("Root Folder:", self.root_input, browse_root)

        # JSON config editor
        self.schema_editor = QtWidgets.QPlainTextEdit()
        self.schema_editor.setMinimumHeight(400)
        self.schema_editor.setPlaceholderText("Paste or write your JSON configuration here...")
        self.schema_editor.setPlainText(json.dumps(CONFIG_TEMPLATE, indent=2))
        self.layout.addWidget(QtWidgets.QLabel("Configuration Schema (JSON):"))
        self.layout.addWidget(self.schema_editor)

        # Number of samples
        self.num_samples = QtWidgets.QSpinBox()
        self.num_samples.setRange(0, int(1e6))
        self.num_samples.setSingleStep(1000)
        self.num_samples.setValue(0)
        self.add_labeled_row("Number of samples (0 = all):", self.num_samples)

        # Parameter legend
        legend = QtWidgets.QLabel(
            "root_dir: folder with input images\n"
            "features_file: file where image embeddings will be saved. If exists, features are loaded from file.\n"
            "Multi-class annotation is supported"
        )
        legend.setStyleSheet("color: gray; font-size: 10pt;")
        self.layout.addWidget(legend)

        # OK / Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        # History section
        self.layout.addWidget(QtWidgets.QLabel("Configuration History:"))
        self.history_list = QtWidgets.QListWidget()
        self.history_list.setMinimumHeight(400)
        self.history_list.setAlternatingRowColors(True)
        self.history_list.setStyleSheet("QListWidget::item { margin: 5px; }")
        self.history_list.itemClicked.connect(self.apply_history_item)
        self.layout.addWidget(self.history_list)
        self.load_history()

    def add_labeled_row(self, label_text, widget, extra_widget=None):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(label_text))
        layout.addWidget(widget)
        if extra_widget:
            layout.addWidget(extra_widget)
        self.layout.addLayout(layout)

    def browse_features(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Feature File")
        if fname:
            self.features_input.setText(fname)

    def browse_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Root Folder")
        if folder:
            self.root_input.setText(folder)

    def load_history(self):
        self.history_data = []
        self.history_list.clear()
        if os.path.exists(CONFIG_HISTORY_PATH):
            try:
                with open(CONFIG_HISTORY_PATH, 'r') as f:
                    self.history_data = json.load(f)

                for idx, item in enumerate(self.history_data):
                    feature = item.get("__features", "N/A")
                    root = item.get("__root", "N/A")
                    keys = [k for k in item.keys() if not k.startswith("__")]

                    text = f"{feature}\n{root}\n{', '.join(keys)}"
                    list_item = QtWidgets.QListWidgetItem(text)
                    list_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
                    list_item.setSizeHint(QSize(0, 60))
                    self.history_list.addItem(list_item)

            except Exception as e:
                print(f"Could not load history: {e}")


    def apply_history_item(self, item):
        index = self.history_list.row(item)
        schema = self.history_data[index]
        self.schema_editor.setPlainText(json.dumps({k: v for k, v in schema.items() if not k.startswith("__")}, indent=2))
        self.features_input.setText(schema.get("__features", ""))
        self.root_input.setText(schema.get("__root", ""))

    def save_to_history(self, schema):
        full_entry = dict(schema)
        full_entry["__features"] = self.features_input.text()
        full_entry["__root"] = self.root_input.text()
        history = []
        if os.path.exists(CONFIG_HISTORY_PATH):
            try:
                with open(CONFIG_HISTORY_PATH, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass
        if full_entry in history:
            history.remove(full_entry)
        history.insert(0, full_entry)
        history = history[:MAX_HISTORY]
        with open(CONFIG_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    def validate_and_accept(self):
        try:
            schema = json.loads(self.schema_editor.toPlainText())
            self.schema_json = schema
        except json.JSONDecodeError:
            QtWidgets.QMessageBox.critical(self, "Invalid JSON", "The schema must be a valid JSON object.")
            return
        if not self.features_input.text().strip() or not self.root_input.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing Fields", "Features file and root folder are required.")
            return
        self.save_to_history(schema)
        self.accept()

    def get_values(self):
        return (
            self.features_input.text(),
            self.root_input.text(),
            self.schema_json,
            int(self.num_samples.value())
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # dialog = ConfigDialog("features.pt", "images/")
    dialog = ConfigDialog()
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        features_file, root_folder, schema, num_samples = dialog.get_values()
        print("Features:", features_file)
        print("Root:", root_folder)
        print("Schema:", schema)
        print("Num Samples:", num_samples)