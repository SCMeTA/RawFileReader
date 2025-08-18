import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, \
    QMessageBox, QCheckBox, QSpinBox, QHBoxLayout

from RawFileExacter import convert_folder_to_mzml

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class ConverterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 布局
        layout = QVBoxLayout()

        self.input_folder_label = QLabel("Input Folder:")
        layout.addWidget(self.input_folder_label)
        self.input_folder_edit = QLineEdit(self)
        layout.addWidget(self.input_folder_edit)
        self.input_folder_button = QPushButton("Select Input Folder", self)
        self.input_folder_button.clicked.connect(self.select_input_folder)
        layout.addWidget(self.input_folder_button)

        self.output_folder_label = QLabel("Output Folder:")
        layout.addWidget(self.output_folder_label)
        self.output_folder_edit = QLineEdit(self)
        layout.addWidget(self.output_folder_edit)
        self.output_folder_button = QPushButton("Select Output Folder", self)
        self.output_folder_button.clicked.connect(self.select_output_folder)
        layout.addWidget(self.output_folder_button)

        checkbox_layout = QHBoxLayout()
        # Put the checkboxes in one row
        self.include_ms2_checkbox = QCheckBox("Include MS2", self)
        self.include_blank_checkbox = QCheckBox("Include Blank", self)
        checkbox_layout.addWidget(self.include_ms2_checkbox)
        checkbox_layout.addWidget(self.include_blank_checkbox)

        layout.addLayout(checkbox_layout)

        self.filter_threshold_label = QLabel("Filter Threshold:")
        layout.addWidget(self.filter_threshold_label)
        self.filter_threshold_spinbox = QSpinBox(self)
        self.filter_threshold_spinbox.setMinimum(0)
        self.filter_threshold_spinbox.setMaximum(10000)
        self.filter_threshold_spinbox.setValue(100)
        layout.addWidget(self.filter_threshold_spinbox)

        self.convert_button = QPushButton("Start Conversion", self)
        self.convert_button.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_button)

        # 设置窗口布局和标题
        self.setLayout(layout)
        self.setWindowTitle('File Converter')
        self.setGeometry(300, 300, 400, 300)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        self.input_folder_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def start_conversion(self):
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        include_ms2 = self.include_ms2_checkbox.isChecked()
        include_blank = self.include_blank_checkbox.isChecked()
        filter_threshold = self.filter_threshold_spinbox.value()

        if not input_folder or not output_folder:
            QMessageBox.critical(self, "Error", "Please provide input and output folder paths.")
            return

        convert_folder_to_mzml(input_folder, output_folder, include_ms2, filter_threshold, include_blank)
        QMessageBox.information(self, "Conversion Completed", "Conversion Completed.")


if __name__ == '__main__':
    logger.info("Starting GUI")
    app = QApplication(sys.argv)
    ex = ConverterApp()
    ex.show()
    sys.exit(app.exec_())