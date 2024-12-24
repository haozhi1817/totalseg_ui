import os
import sys

import numpy as np
import nibabel as nib

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QScrollBar,
    QSizePolicy,
    QLineEdit,
    QSlider,
    QMessageBox,
    QGroupBox,
    QAction,
    QRadioButton,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

from infer import Infer

SLIDER_MIN = -2048
SLIDER_MAX = 4096
OPACITY_MIN = 0
OPACITY_MAX = 100
HU_MIN = -140
HU_MAX = 210
NUM_LABELS = 105


def generate_distinct_colors(num_labels):
    colors = []
    for i in range(num_labels):
        r = (i * 67) % 256  # 使用质数生成避免颜色周期性
        g = (i * 97) % 256
        b = (i * 137) % 256
        colors.append([r, g, b])
    return np.array(colors)


class ImageLabel(QLabel):
    """
    自定义 QLabel，用于绘制十字交叉线
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_position = None

    def set_click_position(self, position):
        """
        设置鼠标点击位置
        """
        self.click_position = position

    def paintEvent(self, event):
        """
        重载 paintEvent，绘制十字交叉线
        """
        super().paintEvent(event)

        if self.click_position is not None:
            painter = QPainter(self)
            pen = QPen(Qt.blue, 2)  # 蓝色，线宽为 2
            painter.setPen(pen)

            # 绘制水平线
            painter.drawLine(
                0, self.click_position.y(), self.width(), self.click_position.y()
            )
            # 绘制垂直线
            painter.drawLine(
                self.click_position.x(), 0, self.click_position.x(), self.height()
            )
            painter.end()


class NiiSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nii.gz Segmentation Tool")
        self.setGeometry(800, 800, 1800, 1600)

        self.color_map = generate_distinct_colors(NUM_LABELS)
        self.segmented_data = None
        self.show_segmentation_enabled = False
        self.load_file_default_path = ""
        self.model_path_default_path = ""
        self.save_path_default_path = ""

        self.initUI()

        # Variables to hold paths and data
        self.nii_file_path = None
        self.model_path = None
        self.save_path = None
        self.current_slice_index = 0  # Current slice index for viewing
        self.nii_data = None  # Loaded Nii.gz data

    def initUI(self):
        # Menubar
        menubar = self.menuBar()
        load_file_menu = menubar.addMenu("Load File")
        load_file_action = QAction("Load File", self)
        load_file_action.triggered.connect(self.load_file)
        load_file_menu.addAction(load_file_action)

        model_path_menu = menubar.addMenu("Model Path")
        model_path_action = QAction("Model Path", self)
        model_path_action.triggered.connect(self.select_model_path)
        model_path_menu.addAction(model_path_action)

        save_path_menu = menubar.addMenu("Save Path")
        save_path_action = QAction("Save Path", self)
        save_path_action.triggered.connect(self.select_save_path)
        save_path_menu.addAction(save_path_action)

        # Layouts
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()  # Layout for image and scrollbar

        # Image display
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)  # Ensure scaling of the image

        # Scroll bar for selecting slices
        self.slice_scrollbar = QScrollBar(Qt.Vertical)
        self.slice_scrollbar.setEnabled(False)
        self.slice_scrollbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slice_scrollbar.valueChanged.connect(self.update_slice_from_scrollbar)

        # Label for slice information
        self.slice_info_label = QLabel("")
        self.slice_info_label.setAlignment(Qt.AlignCenter)

        # Right-side layout (scroll bar and slice info)
        scrollbar_layout = QVBoxLayout()
        scrollbar_layout.addWidget(self.slice_scrollbar)
        scrollbar_layout.addWidget(self.slice_info_label)

        # Window-Center Window-Width Layout
        left_layout = QVBoxLayout()
        normalize_group = QGroupBox("Normalize")
        normalize_layout = QVBoxLayout()
        center_width_layout = QHBoxLayout()
        center_label = QLabel("Window center")
        self.center_input = QLineEdit()
        self.center_input.setText("35")
        width_label = QLabel("Window width")
        self.width_input = QLineEdit()
        self.width_input.setText("350")
        center_width_layout.addWidget(center_label)
        center_width_layout.addWidget(self.center_input)
        center_width_layout.addWidget(width_label)
        center_width_layout.addWidget(self.width_input)
        normalize_layout.addLayout(center_width_layout)
        apply_button = QPushButton("Apply")
        normalize_layout.addWidget(apply_button)
        apply_button.clicked.connect(self.update_image_display)

        x_min_layout = QHBoxLayout()
        x_max_layout = QHBoxLayout()
        self.x_min_slider = QSlider(Qt.Horizontal)
        self.x_min_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.x_min_slider.setValue(HU_MIN)
        self.x_max_slider = QSlider(Qt.Horizontal)
        self.x_max_slider.setRange(SLIDER_MIN, SLIDER_MAX)
        self.x_max_slider.setValue(HU_MAX)
        self.x_min_slider.valueChanged.connect(self.update_window_labels)
        self.x_max_slider.valueChanged.connect(self.update_window_labels)
        self.x_min_slider.valueChanged.connect(self.enforce_x_min_constraints)
        self.x_max_slider.valueChanged.connect(self.enforce_x_max_constraints)
        self.x_min_label = QLabel(f"X Min: {HU_MIN}")
        self.x_max_label = QLabel(f"X Max: {HU_MAX}")
        x_min_layout.addWidget(self.x_min_label)
        x_min_layout.addWidget(self.x_min_slider)
        x_max_layout.addWidget(self.x_max_label)
        x_max_layout.addWidget(self.x_max_slider)
        normalize_layout.addLayout(x_min_layout)
        normalize_layout.addLayout(x_max_layout)
        normalize_group.setLayout(normalize_layout)
        left_layout.addWidget(normalize_group)

        # oppacity Layout
        mask_group = QGroupBox("Mask")
        mask_layout = QVBoxLayout()
        opacity_layout = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(OPACITY_MIN, OPACITY_MAX)
        self.opacity_slider.setValue((OPACITY_MIN + OPACITY_MAX) // 2)
        self.opacity_slider.valueChanged.connect(self.update_mask_display)
        opacity_label = QLabel("Opacity")
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider)

        self.show_button = QPushButton("Show")
        self.show_button.clicked.connect(self.show_segmentation)
        self.show_button.setEnabled(False)

        mask_layout.addLayout(opacity_layout)
        mask_layout.addWidget(self.show_button)
        mask_group.setLayout(mask_layout)
        left_layout.addWidget(mask_group)

        # Info Layout
        self.info_group = QGroupBox("Info")
        pos_layout = QHBoxLayout()
        value_layout = QHBoxLayout()
        info_layout = QVBoxLayout()
        self.hu_value_label = QLabel("HU Value: N/A")
        self.mask_value_label = QLabel("Mask Value: N/A")
        self.pos_x_label = QLabel("X: N/A")
        self.pos_y_label = QLabel("Y: N/A")
        self.pos_z_label = QLabel("Z: N/A")
        pos_layout.addWidget(self.pos_x_label)
        pos_layout.addWidget(self.pos_y_label)
        pos_layout.addWidget(self.pos_z_label)
        info_layout.addLayout(pos_layout)
        value_layout.addWidget(self.hu_value_label)
        value_layout.addWidget(self.mask_value_label)
        info_layout.addLayout(value_layout)
        self.info_group.setLayout(info_layout)
        left_layout.addWidget(self.info_group)

        # Model Layout
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.fast_button = QRadioButton("Fast")
        self.slow_button = QRadioButton("Slow")
        self.fast_button.setChecked(True)
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.fast_button)
        self.mode_button_group.addButton(self.slow_button)
        mode_layout.addWidget(self.fast_button)
        mode_layout.addWidget(self.slow_button)

        device_layout = QHBoxLayout()
        self.gpu_button = QRadioButton("GPU")
        self.cpu_button = QRadioButton("CPU")
        self.gpu_button.setChecked(True)
        self.device_button_group = QButtonGroup()
        self.device_button_group.addButton(self.gpu_button)
        self.device_button_group.addButton(self.cpu_button)
        device_layout.addWidget(self.gpu_button)
        device_layout.addWidget(self.cpu_button)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_segmentation)
        self.run_button.setEnabled(False)

        model_layout.addLayout(mode_layout)
        model_layout.addLayout(device_layout)
        model_layout.addWidget(self.run_button)
        model_group.setLayout(model_layout)

        left_layout.addWidget(model_group)

        # Add image and scrollbar to image layout
        image_layout.addLayout(left_layout, stretch=1)
        image_layout.addWidget(self.image_label, stretch=5)
        image_layout.addLayout(scrollbar_layout, stretch=1)

        # Add layouts to main layout
        main_layout.addLayout(image_layout)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Nii.gz File",
            self.load_file_default_path,
            "Nii.gz Files (*.nii.gz)",
        )
        if file_path:
            self.load_file_default_path = file_path
            self.nii_file_path = file_path
            self.nii_data = nib.load(file_path).get_fdata().transpose(1, 0, 2)
            self.current_slice_index = (
                self.nii_data.shape[2] // 2
            )  # Initialize to middle slice
            self.slice_scrollbar.setEnabled(True)
            self.slice_scrollbar.setMaximum(self.nii_data.shape[2] - 1)
            self.slice_scrollbar.setValue(self.current_slice_index)
            self.segmented_data = None
            self.display_image(self.nii_data, slice_index=self.current_slice_index)
            self.update_slice_info(self.current_slice_index)
            self.show_segmentation_enabled = False

    def select_model_path(self):
        self.model_path = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", self.model_path_default_path
        )
        self.model_path_default_path = self.model_path
        self.run_button.setEnabled(True)

    def select_save_path(self):
        self.save_path = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", self.save_path_default_path
        )
        self.save_path_default_path = self.save_path
        if self.nii_file_path and self.save_path:
            output_filename = (
                os.path.basename(self.nii_file_path).split(".")[0] + "_mask.nii.gz"
            )
            output_path = os.path.join(self.save_path, output_filename)
            if os.path.exists(output_path):
                self.show_button.setEnabled(True)

    def run_segmentation(self):
        if not self.nii_file_path or not self.model_path or not self.save_path:
            self.statusBar().showMessage(
                "Please load file, set model path, and save path first.", 5000
            )
            return

        self.run_button.setEnabled(False)
        self.show_button.setEnabled(False)
        self.statusBar().showMessage("Running segmentation...")

        # Call the external Segmentation function
        input_path = self.nii_file_path
        input_filename = os.path.basename(input_path).split(".")[0]
        output_filename = os.path.splitext(input_filename)[0] + "_mask.nii.gz"
        output_path = os.path.join(self.save_path, output_filename)
        tmp_folder = os.path.join(self.save_path, "tmp")
        self.Segmentation(input_path, self.model_path, tmp_folder, output_path)

        self.statusBar().showMessage(
            f"Segmentation completed and saved to {output_path}", 5000
        )
        self.run_button.setEnabled(True)
        self.show_button.setEnabled(True)

    def Segmentation(self, input_path, model_folder, tmp_folder, output_path):
        # Replace this with your actual segmentation logic
        mode = self.mode_button_group.checkedButton().text()
        devide = self.device_button_group.checkedButton().text()
        if mode == "Fast":
            if_fast = True
        else:
            if_fast = False

        if devide == "GPU":
            all_in_gpu = True
        else:
            all_in_gpu = False

        infer = Infer(
            nii_input_path=input_path,
            model_folder=model_folder,
            tmp_folder=tmp_folder,
            nii_out_path=output_path,
            if_fast=if_fast,
            split_margin=20,
            all_in_gpu=all_in_gpu,
            mix_precision=True,
        )
        infer()

    def show_segmentation(self):
        if self.nii_file_path and self.save_path:
            output_filename = (
                os.path.basename(self.nii_file_path).split(".")[0] + "_mask.nii.gz"
            )
            self.mask_path = os.path.join(self.save_path, output_filename)
            if os.path.exists(self.mask_path):
                if not self.show_segmentation_enabled:
                    self.segmented_data = (
                        nib.load(self.mask_path).get_fdata().transpose(1, 0, 2)
                    )
                    self.segmented_color_data = self.color_map[
                        self.segmented_data.astype("uint8")
                    ]
                    self.display_image(
                        self.nii_data,
                        slice_index=self.current_slice_index,
                    )
                    self.show_segmentation_enabled = True
                else:
                    self.show_segmentation_enabled = False
                    self.segmented_data = None
                    self.display_image(
                        self.nii_data,
                        slice_index=self.current_slice_index,
                    )
            else:
                self.show_button.setEnabled(False)
                QMessageBox.critical(self, "Error", "Segmentation file not found.")
                return
        else:
            print("Please select a NIfTI file and a save path first.")
            self.show_button.setEnabled(False)
            QMessageBox.critical(self, "Error", "Segmentation file not found.")
            return

    def display_image(self, data, slice_index=0):
        if data is not None:
            slice_data = data[:, :, slice_index]
            x_min = self.x_min_slider.value()
            x_max = self.x_max_slider.value()
            slice_data = np.clip(slice_data, x_min, x_max)
            slice_data = (slice_data - x_min) / (x_max - x_min) * 255
            slice_data = slice_data.astype(np.uint8)
            slice_data = np.tile(slice_data[..., None], (1, 1, 3))
            height, width = slice_data.shape[:2]
            bytes_per_line = width

            if self.segmented_data is not None:
                opacity = self.opacity_slider.value() / 100
                mask_slice = self.segmented_color_data[:, :, slice_index, :]
                mask = np.tile((mask_slice.sum(-1) > 0)[..., None], (1, 1, 3))
                slice_data = (
                    slice_data * mask * (1 - opacity)
                    + slice_data * (1 - mask)
                    + mask_slice * mask * opacity
                ).astype("uint8")

            image = QImage(
                slice_data.tobytes(),
                width,
                height,
                bytes_per_line * 3,
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(image)

            # Scale pixmap to fit label while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("Failed to load image.")

    def update_image_display(
        self,
    ):
        self.window_center = int(self.center_input.text())
        self.window_width = int(self.width_input.text())
        x_min = self.window_center - self.window_width // 2
        x_max = x_min + self.window_width
        self.x_min_slider.setValue(x_min)
        self.x_max_slider.setValue(x_max)
        self.display_image(self.nii_data, slice_index=self.current_slice_index)

    def update_mask_display(self):
        self.display_image(self.nii_data, slice_index=self.current_slice_index)

    def update_slice_from_scrollbar(self, value):
        self.current_slice_index = value
        self.display_image(self.nii_data, slice_index=self.current_slice_index)
        self.update_slice_info(value)

    def update_slice_info(self, slice_index):
        if self.nii_data is not None:
            self.slice_info_label.setText(
                f"Slice {slice_index + 1}/{self.nii_data.shape[2]}"
            )

    def update_window_labels(
        self,
    ):
        x_min_value = self.x_min_slider.value()
        x_max_value = self.x_max_slider.value()
        self.x_min_label.setText(f"X Min: {x_min_value}")
        self.x_max_label.setText(f"X Max: {x_max_value}")

    def enforce_x_min_constraints(self):
        """
        动态限制 x_min 的行为。
        """
        x_max_value = self.x_max_slider.value()
        x_min_value = self.x_min_slider.value()

        # 如果 x_min 超过了 x_max，强制回退
        if x_min_value > x_max_value:
            self.x_min_slider.setValue(x_max_value)
        self.display_image(self.nii_data, slice_index=self.current_slice_index)

    def enforce_x_max_constraints(self):
        """
        动态限制 x_min 的行为。
        """
        x_max_value = self.x_max_slider.value()
        x_min_value = self.x_min_slider.value()

        # 如果 x_min 超过了 x_max，强制回退
        if x_min_value > x_max_value:
            self.x_max_slider.setValue(x_min_value)
        self.display_image(self.nii_data, slice_index=self.current_slice_index)

    def resizeEvent(self, event):
        if self.nii_data is not None:
            # Redraw current slice with new label size
            self.display_image(self.nii_data, slice_index=self.current_slice_index)
        super().resizeEvent(event)

    def wheelEvent(self, event):
        if self.nii_data is not None:
            delta = event.angleDelta().y() // 120  # Mouse wheel direction
            new_slice_index = self.current_slice_index + delta

            # Clamp slice index to valid range
            new_slice_index = max(0, min(self.nii_data.shape[2] - 1, new_slice_index))

            if new_slice_index != self.current_slice_index:
                self.current_slice_index = new_slice_index
                self.slice_scrollbar.setValue(self.current_slice_index)  # Sync slider
                self.display_image(self.nii_data, slice_index=self.current_slice_index)
                self.update_info_label()

    def mousePressEvent(self, event):
        """
        捕获鼠标点击事件
        """
        if event.button() == Qt.LeftButton and self.nii_data is not None:
            # 获取鼠标点击位置
            pos = self.image_label.mapFromGlobal(event.globalPos())
            if not self.image_label.pixmap():
                return  # 没有加载图像时直接返回

            # 获取实际显示的图像大小
            pixmap_size = self.image_label.pixmap().size()
            label_size = self.image_label.size()
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2
            if (
                x_offset <= pos.x() <= x_offset + pixmap_size.width()
                and y_offset <= pos.y() <= y_offset + pixmap_size.height()
            ):
                # 映射到图像像素坐标
                img_x = (
                    (pos.x() - x_offset) * self.nii_data.shape[0] / pixmap_size.width()
                )
                img_y = (
                    (pos.y() - y_offset) * self.nii_data.shape[1] / pixmap_size.height()
                )

                self.img_x = int(img_x)
                self.img_y = int(img_y)

                # 检查是否越界
                if (
                    0 <= img_x < self.nii_data.shape[0]
                    and 0 <= img_y < self.nii_data.shape[1]
                ):
                    # 更新 HU 值和分割标签

                    self.click_position = QPoint(pos.x(), pos.y())
                    self.image_label.set_click_position(self.click_position)  # 更新点击位置
                    self.image_label.repaint()
                    self.pos_x_label.setText(f"X: {self.img_y}")
                    self.pos_y_label.setText(f"Y: {self.img_x}")
                    self.pos_z_label.setText(f"Z: {self.current_slice_index}")
                    hu_value = self.nii_data[
                        self.img_y, self.img_x, self.current_slice_index
                    ]
                    self.hu_value_label.setText(f"HU Value: {hu_value:.2f}")
                    if self.segmented_data is not None:
                        label_value = self.segmented_data[
                            self.img_y, self.img_x, self.current_slice_index
                        ]
                        self.mask_value_label.setText(
                            f"Segmentation Label: {int(label_value)}"
                        )
                    else:
                        self.mask_value_label.setText("Segmentation Label: N/A")

    def update_info_label(
        self,
    ):
        try:
            self.pos_x_label.setText(f"X: {self.img_y}")
            self.pos_y_label.setText(f"Y: {self.img_x}")
            self.pos_z_label.setText(f"Z: {self.current_slice_index}")
            hu_value = self.nii_data[self.img_y, self.img_x, self.current_slice_index]
            self.hu_value_label.setText(f"HU Value: {hu_value:.2f}")
            if self.segmented_data is not None:
                label_value = self.segmented_data[
                    self.img_y, self.img_x, self.current_slice_index
                ]
                self.mask_value_label.setText(f"Segmentation Label: {int(label_value)}")
            else:
                self.mask_value_label.setText("Segmentation Label: N/A")
        except:
            self.pos_x_label.setText(f"X: N/A")
            self.pos_y_label.setText(f"Y: N/A")
            self.pos_z_label.setText(f"Z: N/A")
            self.hu_value_label.setText(f"HU Value: N/A")
            self.mask_value_label.setText("Segmentation Label: N/A")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NiiSegmentationApp()
    window.show()
    sys.exit(app.exec_())
