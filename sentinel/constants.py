"""
Configuration constants for the Sentinel application.
"""

import os

# File and directory paths
SETTINGS_FILE = "settings.json"
CAPTURE_DIR = "captured_images"

# Camera and motion detection settings
HISTORY_LENGTH = 500  # History length for background subtractor
VAR_THRESHOLD = 50    # Variance threshold for background subtractor
DETECT_SHADOWS = True # Whether to detect shadows
MIN_CONTOUR_AREA = 700  # Minimum contour area to consider as motion
FPS = 30              # Target frames per second
FRAME_INTERVAL = int(1000 / FPS)  # Timer interval in milliseconds

# Motion alert settings
MOTION_TIMEOUT = 10.0  # Seconds between motion alerts

# Settings management
SAVE_DEBOUNCE_INTERVAL = 1.0  # Seconds between settings save operations

# UI settings
CONTROL_PANEL_WIDTH = 300  # Width of the control panel in pixels
DEFAULT_WINDOW_SIZE = (800, 600)  # Default window size (width, height)
DEFAULT_FEED_SIZE = (640, 480)    # Default camera feed size (width, height)

# Audio settings
DEFAULT_SPEECH_SPEED = 150  # Default speech rate
DEFAULT_VOLUME = 1.0       # Default volume (0.0 to 1.0)

# Video recording settings
VIDEO_DIR = "recorded_videos"  # Directory to store video recordings
RECORDING_DURATION = 10.0  # Duration of recordings in seconds
RECORDING_FPS = 20  # Frames per second for recordings
VIDEO_CODEC = "XVID"  # Codec to use for recordings (XVID is widely supported)
MOTION_COOLDOWN = 2.0  # Seconds to wait before starting a new recording after motion

# Application stylesheet - Dark theme with cyan accents
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2E2E2E;
    color: #00FFFF;
}
QGroupBox {
    border: 1px solid #00FFFF;
    border-radius: 5px;
    margin-top: 10px;
    color: #00FFFF;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 3px;
    color: #00FFFF;
}
QPushButton, QRadioButton {
    background-color: #00FFFF;
    color: #2E2E2E;
    border: none;
    padding: 5px;
    border-radius: 3px;
}
QPushButton:hover, QRadioButton:hover {
    background-color: #00CCCC;
}
QComboBox, QSlider, QLabel {
    color: #00FFFF;
}
QComboBox {
    background-color: #3E3E3E;
    border: 1px solid #00FFFF;
    padding: 2px;
}
QComboBox::drop-down {
    border: none;
}
QSlider::groove:horizontal {
    background: #3E3E3E;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #00FFFF;
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -4px 0;
}
"""
