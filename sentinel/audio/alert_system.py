"""
Audio alert system for Sentinel.
"""

import time
from typing import Dict, Any, List, Optional
import pyttsx3

from sentinel.constants import (
    DEFAULT_SPEECH_SPEED,
    DEFAULT_VOLUME,
    MOTION_TIMEOUT
)


class AudioAlertSystem:
    """Handles text-to-speech alerts for motion detection."""
    
    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize the audio alert system.
        
        Args:
            settings: Application settings dictionary
        """
        self.last_alert_time = 0
        self.tts_engine = None
        self.voices = []
        self.voice_ids = []
        
        try:
            self.tts_engine = pyttsx3.init()
            self.voices = self.tts_engine.getProperty("voices")
            self.voice_ids = [voice.id for voice in self.voices]
            
            # Initialize TTS properties from settings
            self.tts_engine.setProperty("rate", settings.get("speech_speed", DEFAULT_SPEECH_SPEED))
            self.tts_engine.setProperty("volume", settings.get("volume", DEFAULT_VOLUME))
            
            # Set voice if available, otherwise use first voice
            voice_id = settings.get("voice_id", self.voice_ids[0] if self.voice_ids else None)
            if voice_id and voice_id in self.voice_ids:
                self.tts_engine.setProperty("voice", voice_id)
            elif self.voice_ids:
                self.tts_engine.setProperty("voice", self.voice_ids[0])
                
            print("TTS engine initialized successfully")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
    
    def trigger_alert(self, message: str = "Motion detected!") -> bool:
        """
        Trigger a text-to-speech alert if the timeout has passed.
        
        Args:
            message: The message to speak
            
        Returns:
            True if alert was triggered, False otherwise
        """
        current_time = time.time()
        if not self.tts_engine or (current_time - self.last_alert_time) <= MOTION_TIMEOUT:
            return False
            
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            self.last_alert_time = current_time
            return True
        except Exception as e:
            print(f"Error triggering alert: {e}")
            return False
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update TTS engine settings.
        
        Args:
            settings: New settings to apply
        """
        if not self.tts_engine:
            return
            
        try:
            if "speech_speed" in settings:
                self.tts_engine.setProperty("rate", settings["speech_speed"])
                
            if "volume" in settings:
                self.tts_engine.setProperty("volume", settings["volume"])
                
            if "voice_id" in settings and settings["voice_id"] in self.voice_ids:
                self.tts_engine.setProperty("voice", settings["voice_id"])
        except Exception as e:
            print(f"Error updating TTS settings: {e}")
    
    def cleanup(self) -> None:
        """Clean up TTS engine resources."""
        if not self.tts_engine:
            return
            
        try:
            print("Stopping TTS engine...")
            self.tts_engine.stop()
            if hasattr(self.tts_engine, '_inLoop') and self.tts_engine._inLoop:
                self.tts_engine.endLoop()
            self.tts_engine = None
        except Exception as e:
            print(f"Error cleaning up TTS engine: {e}")
