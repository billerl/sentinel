"""
Main window for the Sentinel application.
"""

import sys
import os
import cv2
import numpy as np
import subprocess  # Add this for opening videos with system player
from typing import List, Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QSlider,
    QLabel,
    QGroupBox,
    QRadioButton,
    QFileDialog,
)
from PySide6.QtGui import QImage, QPixmap, QIcon, QCloseEvent, QResizeEvent
from PySide6.QtCore import Qt, QTimer

from sentinel.constants import (
    CONTROL_PANEL_WIDTH,
    FRAME_INTERVAL,
    CAPTURE_DIR,
    VIDEO_DIR,
    STYLESHEET,
)
from sentinel.detection import MotionDetector
from sentinel.audio import AudioAlertSystem
from sentinel.settings import SettingsManager


class MainWindow(QMainWindow):
    """
    Main window for the Sentinel motion detection application.
    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        motion_detector: MotionDetector,
        audio_system: AudioAlertSystem,
    ):
        """
        Initialize the main window.

        Args:
            settings_manager: Settings manager for the application
            motion_detector: Motion detector for camera processing
            audio_system: Audio alert system for TTS alerts
        """
        super().__init__()

        # Store component references
        self.settings_manager = settings_manager
        self.motion_detector = motion_detector
        self.audio_system = audio_system

        # Set up timer for camera feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)

        # Set up UI
        self.setup_ui()

        # Initial camera search and setup
        self.search_cameras()
        self.show_blank_frame()
        if self.settings_manager.get("feed_active", False):
            self.toggle_feed()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        # Set window properties
        self.setWindowTitle("Sentinel - Motion Detection")
        icon = QIcon("sentinel.ico")
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setWindowIconText("Sentinel")

        # Set window size from settings
        window_size = self.settings_manager.get("window_size", (800, 600))
        self.setGeometry(100, 100, window_size[0], window_size[1])

        # Create main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create UI components
        self.setup_control_panel()
        self.setup_camera_feed()

        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)

    def setup_control_panel(self) -> None:
        """Set up the left control panel with all controls."""
        # Create control panel container
        self.control_widget = QWidget()
        self.control_widget.setFixedWidth(CONTROL_PANEL_WIDTH)
        self.control_layout = QVBoxLayout(self.control_widget)
        self.main_layout.addWidget(self.control_widget)

        # Add control groups
        self.setup_camera_controls()
        self.setup_speech_controls()
        self.setup_orientation_controls()

        # Add status label
        self.status_label = QLabel("Camera: Disconnected")
        self.control_layout.addWidget(self.status_label)

    def setup_camera_controls(self) -> None:
        """Set up camera selection and control buttons."""
        # Create camera group
        self.camera_group = QGroupBox("Select Camera:")
        self.camera_group.setStyleSheet("QGroupBox::title { color: #00FFFF; }")
        self.camera_layout = QVBoxLayout()

        # Camera selection dropdown - DON'T initialize with any cameras yet
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self.on_camera_select)
        self.camera_layout.addWidget(self.camera_combo)

        # Camera control buttons
        self.search_btn = QPushButton("Search Cameras")
        self.search_btn.clicked.connect(self.search_cameras)
        self.camera_layout.addWidget(self.search_btn)

        self.toggle_btn = QPushButton("Camera Offline")
        self.toggle_btn.clicked.connect(self.toggle_feed)
        self.camera_layout.addWidget(self.toggle_btn)

        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.camera_layout.addWidget(self.capture_btn)

        self.recall_btn = QPushButton("Recall Image")
        self.recall_btn.clicked.connect(self.recall_image)
        self.camera_layout.addWidget(self.recall_btn)

        # Add to control panel
        self.camera_group.setLayout(self.camera_layout)
        self.control_layout.addWidget(self.camera_group)

    def setup_speech_controls(self) -> None:
        """Set up speech and audio controls."""
        # Speech speed control
        self.speech_group = QGroupBox("Speech Speed:")
        self.speech_group.setStyleSheet("QGroupBox::title { color: #00FFFF; }")
        self.speech_layout = QVBoxLayout()

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 300)
        self.speed_slider.setValue(self.settings_manager.get("speech_speed", 150))
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.speech_layout.addWidget(self.speed_slider)

        self.speech_group.setLayout(self.speech_layout)
        self.control_layout.addWidget(self.speech_group)

        # Volume control
        self.volume_group = QGroupBox("Volume:")
        self.volume_group.setStyleSheet("QGroupBox::title { color: #00FFFF; }")
        self.volume_layout = QVBoxLayout()

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 10)
        self.volume_slider.setValue(int(self.settings_manager.get("volume", 1.0) * 10))
        self.volume_slider.valueChanged.connect(self.on_volume_change)
        self.volume_layout.addWidget(self.volume_slider)

        self.volume_group.setLayout(self.volume_layout)
        self.control_layout.addWidget(self.volume_group)

        # Voice selection
        self.voice_group = QGroupBox("Voice:")
        self.voice_group.setStyleSheet("QGroupBox::title { color: #00FFFF; }")
        self.voice_layout = QVBoxLayout()

        self.voice_buttons = []
        selected_voice = self.settings_manager.get(
            "voice_id",
            self.audio_system.voice_ids[0] if self.audio_system.voice_ids else None,
        )

        for i, voice in enumerate(self.audio_system.voices):
            voice_name = voice.id.split("_")[-2]  # Shortened name (e.g., "DAVID")
            btn = QRadioButton(voice_name)
            btn.setStyleSheet("font-size: 10px;")
            btn.toggled.connect(
                lambda checked, idx=i: self.on_voice_change(idx, checked)
            )

            if voice.id == selected_voice:
                btn.setChecked(True)

            self.voice_buttons.append(btn)
            self.voice_layout.addWidget(btn)

        self.voice_group.setLayout(self.voice_layout)
        self.control_layout.addWidget(self.voice_group)

    def setup_orientation_controls(self) -> None:
        """Set up camera orientation controls."""
        self.orient_group = QGroupBox("Camera View:")
        self.orient_group.setStyleSheet("QGroupBox::title { color: #00FFFF; }")
        self.orient_layout = QHBoxLayout()

        self.horiz_btn = QRadioButton("Horizontal")
        self.horiz_btn.toggled.connect(self.on_cam_orient_change)

        self.vert_btn = QRadioButton("Vertical")
        self.vert_btn.toggled.connect(self.on_cam_orient_change)

        self.orient_layout.addWidget(self.horiz_btn)
        self.orient_layout.addWidget(self.vert_btn)

        # Set initial state from settings
        if self.settings_manager.get("cam_orientation", "horizontal") == "horizontal":
            self.horiz_btn.setChecked(True)
        else:
            self.vert_btn.setChecked(True)

        self.orient_group.setLayout(self.orient_layout)
        self.control_layout.addWidget(self.orient_group)

    def setup_camera_feed(self) -> None:
        """Set up the camera feed display area."""
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)

        # Set size from settings
        feed_size = self.settings_manager.get("feed_size", (640, 480))
        self.camera_feed.setMinimumSize(320, 240)  # Minimum size
        self.camera_feed.resize(feed_size[0], feed_size[1])

        self.main_layout.addWidget(self.camera_feed, stretch=1)

    def search_cameras(self) -> None:
        """Search for available cameras and update the dropdown."""
        camera_list = []

        for i in range(10):  # Check the first 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_list.append(f"Camera {i}")
                    cap.release()
                else:
                    print(f"Camera {i} not available")
            except Exception as e:
                print(f"Error checking camera {i}: {e}")

        # Store the current selection before clearing
        current_index = self.settings_manager.get("camera_index", 0)

        # Update the dropdown
        self.camera_combo.clear()
        if not camera_list:
            self.camera_combo.addItems(["No Cameras Found"])
            self.status_label.setText("Camera: None Detected")
            return

        self.camera_combo.addItems(camera_list)

        # Try to select the camera index from settings
        saved_camera_text = f"Camera {current_index}"

        # Check if the saved camera index is in the list of available cameras
        if saved_camera_text in camera_list:
            self.camera_combo.setCurrentText(saved_camera_text)
            self.motion_detector.camera_index = current_index
            self.status_label.setText(f"Camera: {saved_camera_text} (Disconnected)")
        else:
            # If saved camera is not available, select the first available camera
            # but DON'T save this as a setting yet
            self.camera_combo.setCurrentText(camera_list[0])
            self.status_label.setText(f"Camera: {camera_list[0]} (Disconnected)")

    def toggle_feed(self) -> None:
        """Toggle the camera feed on/off."""
        if not self.motion_detector.feed_active:
            # Turn on camera
            if self.camera_combo.currentText() == "No Cameras Found":
                self.status_label.setText("Camera: None Available")
                return

            camera_index = int(self.camera_combo.currentText().split()[-1])

            if self.motion_detector.start_camera(camera_index):
                self.toggle_btn.setText("Camera Online")
                self.status_label.setText(f"Camera {camera_index}: Connected")
                self.timer.start(FRAME_INTERVAL)

                # Disable the search camera and recall button while the camera is active
                self.search_btn.setEnabled(False)
                self.search_btn.setProperty("inactive", True)
                self.search_btn.setStyleSheet(
                    "background-color: #555555; color: #999999; border: 1px solid #444444;"
                )
                self.recall_btn.setStyleSheet(
                    "background-color: #555555; color: #999999; border: 1px solid #444444;"
                )
                self.recall_btn.setEnabled(False)
                self.recall_btn.setProperty("inactive", True)

                # Update settings
                self.settings_manager.set("camera_index", camera_index)
                self.settings_manager.set("feed_active", True)
            else:
                self.status_label.setText(f"Camera {camera_index}: Failed to Connect")
        else:
            # Turn off camera
            self.motion_detector.stop_camera()
            self.toggle_btn.setText("Camera Offline")
            self.status_label.setText(
                f"Camera {self.settings_manager.get('camera_index', 0)}: Disconnected"
            )
            self.timer.stop()

            # Disable the search camera and recall button while the camera is active
            self.search_btn.setEnabled(True)
            self.search_btn.setProperty("inactive", False)
            self.search_btn.setStyleSheet("")  # Reset to default style
            self.recall_btn.setEnabled(True)
            self.recall_btn.setProperty("inactive", False)
            self.recall_btn.setStyleSheet("")  # Reset to default style

            # Update settings
            self.settings_manager.set("feed_active", False)

            # Show blank frame
            self.show_blank_frame()

    def display_frame(self, frame: np.ndarray) -> None:
        """
        Display a frame in the camera feed.
        
        Args:
            frame: The OpenCV frame to display
        """
        # Convert BGR to RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage and QPixmap
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Display with aspect ratio preservation
        self.camera_feed.setPixmap(pixmap.scaled(self.camera_feed.size(), Qt.KeepAspectRatio))

    def show_blank_frame(self) -> None:
        """Display a blank frame with 'Camera Offline' text."""
        # Get current feed dimensions
        feed_width = self.camera_feed.width()
        feed_height = self.camera_feed.height()

        # Create blank frame
        frame = np.zeros((feed_height, feed_width, 3), dtype=np.uint8)

        # Add text
        text = "Camera Offline"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        # Center the text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (feed_width - text_size[0]) // 2
        text_y = (feed_height + text_size[1]) // 2

        cv2.putText(
            frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
        )

        # Display the frame
        self.display_frame(frame)


    def update_feed(self):
        """Update the camera feed with motion detection."""
        # Get frame with motion detection and recording status
        frame, _, motion_detected, recording_status = self.motion_detector.detect_motion()

        if frame is None:
            self.status_label.setText(
                f"Camera {self.motion_detector.camera_index}: Error Reading Frame"
            )
            self.toggle_feed()  # Turn off camera on error
            return

        # Handle camera orientation
        if self.settings_manager.get("cam_orientation", "horizontal") == "vertical":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Update status based on recording state
        if recording_status == "started":
            self.status_label.setText("Recording started (10 seconds)")
        elif recording_status == "recording":
            # Don't update status during recording to avoid flickering
            pass

        # Trigger alert if motion was detected (only if not already recording)
        if motion_detected:
            self.audio_system.trigger_alert()

        # Display the frame
        self.display_frame(frame)

    def capture_image(self) -> None:
        """Capture and save an image from the camera."""
        result = self.motion_detector.capture_image(
            orientation=self.settings_manager.get("cam_orientation", "horizontal")
        )

        if not result:
            self.status_label.setText("Capture Failed: Camera Offline or Error")
            return

        frame, filename = result

        # Save the image
        try:
            cv2.imwrite(filename, frame)
            self.status_label.setText(f"Image Saved: {filename}")

            # Display the captured frame
            self.display_frame(frame)
        except Exception as e:
            self.status_label.setText(f"Error Saving Image: {str(e)}")

    def recall_image(self) -> None:
        """Open a file dialog to select and display a captured image."""
        try:
            # Create the captures directory if it doesn't exist
            if not os.path.exists(CAPTURE_DIR):
                os.makedirs(CAPTURE_DIR)
                self.status_label.setText("No images found. Capture an image first.")
                return

            # Check if the directory has any images
            image_files = [f for f in os.listdir(CAPTURE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
            if not image_files:
                self.status_label.setText("No images found. Capture an image first.")
                return

            # Open file dialog in the captures directory
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image to Recall",
                CAPTURE_DIR,
                "Images (*.png *.jpg *.jpeg)"
            )

            if not file_path:  # User canceled the dialog
                return

            # Load and display the selected image
            frame = cv2.imread(file_path)

            if frame is None:
                self.status_label.setText(f"Error Loading Image: {file_path}")
                return

            self.display_frame(frame)
            self.status_label.setText(f"Recalled: {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error Recalling Image: {str(e)}")


    def browse_recordings(self):
        """Open file dialog to browse and open recorded videos."""
        try:
            # Create the videos directory if it doesn't exist
            if not os.path.exists(VIDEO_DIR):
                os.makedirs(VIDEO_DIR)
                self.status_label.setText("No videos found.")
                return

            # Check if the directory has any videos
            video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".avi", ".mp4"))]
            if not video_files:
                self.status_label.setText("No videos found.")
                return

            # Open file dialog in the videos directory
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video to Open", VIDEO_DIR, "Videos (*.avi *.mp4)"
            )

            if not file_path:  # User canceled the dialog
                return

            # Open the video with the default system video player
            if sys.platform.startswith("win"):
                os.startfile(file_path)
            elif sys.platform.startswith("darwin"):  # macOS
                subprocess.call(["open", file_path])
            else:  # Linux
                subprocess.call(["xdg-open", file_path])

            self.status_label.setText(f"Opened: {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error opening video: {str(e)}")

    def on_camera_select(self, value: str) -> None:
        """
        Handle camera selection change.

        Args:
            value: The selected camera text
        """
        if value != "No Cameras Found" and value.startswith("Camera"):
            camera_index = int(value.split()[-1])
            self.settings_manager.set("camera_index", camera_index)

    def on_speed_change(self, value: int) -> None:
        """
        Handle speech speed slider change.

        Args:
            value: The new speech speed value
        """
        self.settings_manager.set("speech_speed", value)
        self.audio_system.update_settings({"speech_speed": value})

    def on_volume_change(self, value: int) -> None:
        """
        Handle volume slider change.

        Args:
            value: The new volume value (0-10)
        """
        volume = value / 10.0
        self.settings_manager.set("volume", volume)
        self.audio_system.update_settings({"volume": volume})

    def on_voice_change(self, idx: int, checked: bool) -> None:
        """
        Handle voice selection change.

        Args:
            idx: Index of the selected voice
            checked: Whether the button is checked
        """
        if checked and 0 <= idx < len(self.audio_system.voice_ids):
            voice_id = self.audio_system.voice_ids[idx]
            self.settings_manager.set("voice_id", voice_id)
            self.audio_system.update_settings({"voice_id": voice_id})

    def on_cam_orient_change(self) -> None:
        """Handle camera orientation change."""
        orientation = "horizontal" if self.horiz_btn.isChecked() else "vertical"
        self.settings_manager.set("cam_orientation", orientation)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handle window resize events to save settings.

        Args:
            event: The resize event
        """
        super().resizeEvent(event)

        # Save window size
        self.settings_manager.set("window_size", [self.width(), self.height()])
        self.settings_manager.set(
            "feed_size", [self.camera_feed.width(), self.camera_feed.height()]
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle window close events for cleanup.

        Args:
            event: The close event
        """
        # Clean up resources
        if self.motion_detector.feed_active:
            self.timer.stop()
            self.motion_detector.cleanup()

        # Save final settings
        self.settings_manager.save_settings()

        event.accept()
