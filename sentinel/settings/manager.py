"""
Settings management for Sentinel.
"""

import os
import json
import time
from typing import Dict, Any

from sentinel.constants import (
    SETTINGS_FILE,
    DEFAULT_SPEECH_SPEED,
    DEFAULT_VOLUME,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FEED_SIZE,
    SAVE_DEBOUNCE_INTERVAL,
    MOTION_COOLDOWN
)


class SettingsManager:
    """Handles loading and saving application settings."""

    def __init__(self, settings_file: str = SETTINGS_FILE):
        """
        Initialize the settings manager.
        
        Args:
            settings_file: Path to the settings JSON file
        """
        self.settings_file = settings_file
        self.last_save_time = 0
        self.default_settings = {
            "speech_speed": DEFAULT_SPEECH_SPEED,
            "volume": DEFAULT_VOLUME,
            "voice_id": "",  # Will be set after TTS initialization
            "cam_orientation": "horizontal",
            "camera_index": 0,
            "feed_active": False,
            "window_size": DEFAULT_WINDOW_SIZE,
            "feed_size": DEFAULT_FEED_SIZE,
            # Add new recording settings
            "recording_enabled": True,  # Enable recording by default
            "motion_cooldown": MOTION_COOLDOWN,  # Use value from constants
        }
        self.settings = self.default_settings.copy()
        self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from the settings file.
        
        Returns:
            The loaded settings dictionary
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    loaded_settings = json.load(f)
                    self.settings.update(loaded_settings)
                    print("Settings loaded successfully")
        except (json.JSONDecodeError, PermissionError, OSError) as e:
            print(f"Error loading settings: {e}")
            print("Using default settings")

        return self.settings

    def save_settings(self, settings: Dict[str, Any] = None) -> bool:
        """
        Save settings to the settings file with debouncing.
        
        Args:
            settings: The settings dictionary to save
            
        Returns:
            True if settings were saved, False if debounced
        """
        current_time = time.time()
        if (current_time - self.last_save_time) < SAVE_DEBOUNCE_INTERVAL:
            return False  # Skip saving if called too soon

        if settings:
            self.settings.update(settings)

        try:
            print("Saving settings...")
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f)
            print("Settings saved.")
            self.last_save_time = current_time
            return True
        except (PermissionError, OSError) as e:
            print(f"Error saving settings: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            The setting value or default
        """
        return self.settings.get(key, default)

    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
            save: Whether to save settings after update
        """
        self.settings[key] = value
        if save:
            self.save_settings()
