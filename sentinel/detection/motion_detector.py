"""
Motion detection functionality for Sentinel.
"""

import os
import cv2
import numpy as np
import time
from typing import Tuple, List, Optional

# Import the constants
from sentinel.constants import (
    HISTORY_LENGTH,
    VAR_THRESHOLD,
    DETECT_SHADOWS,
    MIN_CONTOUR_AREA,
    VIDEO_DIR,
    RECORDING_DURATION,
    RECORDING_FPS,
    VIDEO_CODEC,
    MOTION_COOLDOWN,
    CAPTURE_DIR,
)


class MotionDetector:
    """Handles camera access and motion detection."""

    def __init__(self):
        """Initialize the motion detector."""
        self.cap = None
        self.camera_index = 0
        self.feed_active = False
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=HISTORY_LENGTH,
            varThreshold=VAR_THRESHOLD,
            detectShadows=DETECT_SHADOWS
        )
        # Video recording properties
        self.video_writer = None
        self.recording = False
        self.recording_enabled = True  # Add this line
        self.recording_start_time = 0
        self.last_motion_time = 0
        self.frames_recorded = 0
        self.motion_cooldown = MOTION_COOLDOWN  # Add this line

        # Create video directory if it doesn't exist
        if not os.path.exists(VIDEO_DIR):
            os.makedirs(VIDEO_DIR)

    # Add new methods for video recording:

    def start_recording(self, frame_size):
        """
        Start recording a video clip.

        Args:
            frame_size: Tuple of (width, height) for the video frames
        """
        # Only start if not already recording
        if self.recording:
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(VIDEO_DIR, f"motion_{timestamp}.avi")

            # Get the four-character code for the video codec
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

            # Create video writer
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, RECORDING_FPS, frame_size, True  # isColor
            )

            self.recording = True
            self.recording_start_time = time.time()
            self.frames_recorded = 0

            print(f"Started recording: {video_path}")
            return video_path
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            self.video_writer = None
            return None

    def stop_recording(self):
        """Stop the current video recording."""
        if not self.recording or self.video_writer is None:
            return

        try:
            # Release the video writer
            self.video_writer.release()
            self.video_writer = None
            self.recording = False

            duration = time.time() - self.recording_start_time
            print(
                f"Stopped recording. Duration: {duration:.2f}s, Frames: {self.frames_recorded}"
            )

            return True
        except Exception as e:
            print(f"Error stopping recording: {e}")
            self.video_writer = None
            self.recording = False
            return False

    def write_frame(self, frame):
        """
        Write a frame to the current video recording.

        Args:
            frame: The frame to write to the video

        Returns:
            True if the frame was written, False otherwise
        """
        if not self.recording or self.video_writer is None:
            return False

        try:
            self.video_writer.write(frame)
            self.frames_recorded += 1

            # Check if recording duration has elapsed
            current_time = time.time()
            if (current_time - self.recording_start_time) >= RECORDING_DURATION:
                self.stop_recording()

            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False

    def check_recording_status(self, motion_detected, frame):
        """
        Check if recording should start, continue, or stop based on motion detection.
        
        Args:
            motion_detected: Whether motion was detected in the current frame
            frame: The current frame
            
        Returns:
            Tuple of (recording_status, video_path)
            recording_status can be: "started", "recording", "stopped", or None
        """
        current_time = time.time()
        
        # If recording is disabled, stop any ongoing recording and return
        if not self.recording_enabled and self.recording:
            self.stop_recording()
            return "stopped", None
        elif not self.recording_enabled:
            return None, None
        
        # If motion detected and not recording, start recording
        if motion_detected and not self.recording:
            # Check cooldown period
            if (current_time - self.last_motion_time) >= self.motion_cooldown:
                self.last_motion_time = current_time
                
                # Get frame dimensions
                height, width = frame.shape[:2]
                
                # Start recording
                video_path = self.start_recording((width, height))
                if video_path:
                    return "started", video_path
        
        # If recording, write the frame
        if self.recording:
            self.write_frame(frame)
            
            # If motion detected, update the last motion time
            if motion_detected:
                self.last_motion_time = current_time
                
            return "recording", None
            
        return None, None

    def start_camera(self, camera_index: int) -> bool:
        """
        Start the camera at the specified index.
        
        Args:
            camera_index: Index of the camera to use
            
        Returns:
            True if camera started successfully, False otherwise
        """
        if self.feed_active:
            self.stop_camera()

        try:
            self.camera_index = camera_index
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_index}")

            self.feed_active = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def stop_camera(self) -> None:
        """Stop the currently active camera."""
        self.feed_active = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # Modified detect_motion method that also handles recording
    def detect_motion(self):
        """
        Capture a frame and detect motion.

        Returns:
            Tuple containing:
            - The processed frame (or None if no frame could be captured)
            - List of motion rectangles (x, y, w, h)
            - Boolean indicating if motion was detected
            - Recording status message (or None if not recording)
        """
        if not self.feed_active or not self.cap or not self.cap.isOpened():
            return None, [], False, None

        try:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")

            # Make a copy of the original frame for recording
            original_frame = frame.copy()

            # Process frame for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            fgmask = self.fgbg.apply(blurred)
            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter contours by size and draw rectangles
            motion_detected = False
            motion_rects = []

            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                motion_rects.append((x, y, w, h))

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Motion Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                motion_detected = True

            # Handle recording based on motion detection
            recording_status, video_path = self.check_recording_status(
                motion_detected, original_frame
            )

            # Add recording indicator to the frame if recording
            if self.recording:
                elapsed_time = time.time() - self.recording_start_time
                remaining_time = max(0, RECORDING_DURATION - elapsed_time)

                # Add recording indicator with time remaining
                cv2.putText(
                    frame,
                    f"REC {remaining_time:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Add red circle as recording indicator
                cv2.circle(frame, (30, 50), 8, (0, 0, 255), -1)

            return frame, motion_rects, motion_detected, recording_status
        except Exception as e:
            print(f"Error in motion detection: {e}")
            return None, [], False, None

    def cleanup(self):
        """Clean up camera and recording resources."""
        if self.recording:
            self.stop_recording()
        self.stop_camera()

    def capture_image(self, orientation: str = "horizontal") -> Optional[Tuple[np.ndarray, str]]:
        """
        Capture a still image from the camera.
        
        Args:
            orientation: Camera orientation ('horizontal' or 'vertical')
            
        Returns:
            Tuple containing the captured frame and filename, or None if capture failed
        """
        if not self.feed_active or not self.cap or not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture image")

            if orientation == "vertical":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{CAPTURE_DIR}/capture_{timestamp}.png"

            return frame, filename
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up camera resources."""
        self.stop_camera()
