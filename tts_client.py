"""
Text-to-Speech client for WikiTalk
"""
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional

from config import *


class TTSClient:
    def __init__(self):
        self.piper_voice_path = PIPER_VOICE_PATH
        self.piper_config_path = PIPER_CONFIG_PATH
        self.use_piper = self._check_piper_availability()
        
    def _check_piper_availability(self) -> bool:
        """Check if Piper TTS is available"""
        if not os.path.exists(self.piper_voice_path):
            print(f"Piper voice file not found: {self.piper_voice_path}")
            return False
        
        if not os.path.exists(self.piper_config_path):
            print(f"Piper config file not found: {self.piper_config_path}")
            return False
        
        # Check if piper executable is available
        try:
            result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
            if result.returncode != 0:
                print("Piper executable not found in PATH")
                return False
        except FileNotFoundError:
            print("Piper executable not found")
            return False
        
        return True
    
    def speak(self, text: str) -> bool:
        """Convert text to speech and play it"""
        if not text.strip():
            return False
        
        try:
            if self.use_piper:
                return self._speak_with_piper(text)
            else:
                return self._speak_with_say(text)
        except Exception as e:
            print(f"TTS error: {e}")
            return False
    
    def _speak_with_piper(self, text: str) -> bool:
        """Use Piper TTS for speech synthesis"""
        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run piper command
            cmd = [
                'piper',
                '-m', self.piper_voice_path,
                '-c', self.piper_config_path,
                '--output_file', temp_path
            ]
            
            # Pipe text to piper
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                print(f"Piper error: {stderr}")
                return False
            
            # Play the generated audio
            if os.path.exists(temp_path):
                subprocess.run(['afplay', temp_path], check=True)
                os.unlink(temp_path)  # Clean up
                return True
            else:
                print("Piper did not generate audio file")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Piper execution failed: {e}")
            return False
        except Exception as e:
            print(f"Piper error: {e}")
            return False
    
    def _speak_with_say(self, text: str) -> bool:
        """Fallback to macOS 'say' command"""
        try:
            # Limit text length for say command
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            subprocess.run(['say', text], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Say command failed: {e}")
            return False
        except Exception as e:
            print(f"Say error: {e}")
            return False
    
    def test_tts(self):
        """Test TTS functionality"""
        test_text = "Hello, this is a test of the text-to-speech system."
        print(f"Testing TTS with: {test_text}")
        
        if self.use_piper:
            print("Using Piper TTS")
        else:
            print("Using macOS say command")
        
        success = self.speak(test_text)
        if success:
            print("TTS test successful")
        else:
            print("TTS test failed")
        
        return success


if __name__ == "__main__":
    tts = TTSClient()
    tts.test_tts()

