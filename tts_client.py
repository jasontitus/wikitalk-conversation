"""
Text-to-Speech client for WikiTalk
Cross-platform support: Windows, macOS, Linux
"""
import subprocess
import os
import tempfile
import platform
from pathlib import Path
from typing import Optional

from config import *


class TTSClient:
    def __init__(self):
        self.platform = platform.system()  # 'Windows', 'Darwin' (Mac), 'Linux'
        self.piper_voice_path = PIPER_VOICE_PATH
        self.piper_config_path = PIPER_CONFIG_PATH
        self.tts_method = self._select_tts_method()
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for TTS by removing markdown and Wikipedia formatting
        
        Removes:
        - Markdown bold/italic: **text** and *text*
        - Wikipedia citations: [1], [2], etc.
        - Extra newlines and whitespace
        - Special unicode characters that don't speak well
        """
        import re
        
        # Remove markdown bold and italic markers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** → bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic* → italic
        text = re.sub(r'_(.+?)_', r'\1', text)         # _italic_ → italic
        
        # Remove Wikipedia citation brackets [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Replace em-dashes and special dashes with regular dashes for better speech
        text = re.sub(r'—|–|‑', '-', text)
        
        # Remove or normalize other special characters
        text = re.sub(r'•', ' ', text)  # bullet points
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n\n+', ' ', text)
        text = re.sub(r'\n', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def _select_tts_method(self) -> str:
        """Select the best TTS method available for this platform"""
        
        # Try Piper first (works on all platforms)
        if self._check_piper_availability():
            return "piper"
        
        # Platform-specific fallbacks
        if self.platform == "Darwin":  # macOS
            return "say"
        elif self.platform == "Windows":
            # Try Windows SAPI via pyttsx3
            if self._check_pyttsx3_availability():
                return "pyttsx3"
            else:
                return "silent"  # No TTS available
        elif self.platform == "Linux":
            # Try espeak on Linux
            if self._check_espeak_availability():
                return "espeak"
            else:
                return "silent"
        else:
            return "silent"
    
    def _check_piper_availability(self) -> bool:
        """Check if Piper TTS is available"""
        try:
            if not os.path.exists(self.piper_voice_path):
                return False
            
            if not os.path.exists(self.piper_config_path):
                return False
            
            # Check if piper executable is available
            if self.platform == "Windows":
                result = subprocess.run(['where', 'piper'], capture_output=True, text=True)
            else:
                result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
            
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False
    
    def _check_pyttsx3_availability(self) -> bool:
        """Check if pyttsx3 is available (Windows SAPI)"""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False
    
    def _check_espeak_availability(self) -> bool:
        """Check if espeak is available (Linux)"""
        try:
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False
    
    def speak(self, text: str) -> bool:
        """Convert text to speech and play it"""
        if not text.strip():
            return False
        
        # Clean text for better TTS speech
        text = self._clean_text_for_speech(text)
        
        try:
            if self.tts_method == "piper":
                return self._speak_with_piper(text)
            elif self.tts_method == "say":
                return self._speak_with_say(text)
            elif self.tts_method == "pyttsx3":
                return self._speak_with_pyttsx3(text)
            elif self.tts_method == "espeak":
                return self._speak_with_espeak(text)
            else:
                print("⚠️  No TTS method available")
                return False
        except Exception as e:
            print(f"TTS error: {e}")
            return False
    
    def _speak_with_piper(self, text: str) -> bool:
        """Use Piper TTS for speech synthesis (cross-platform)"""
        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run piper command
            cmd = [
                'piper',
                '-m', str(self.piper_voice_path),
                '-c', str(self.piper_config_path),
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
                self._play_audio(temp_path)
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
    
    def _play_audio(self, audio_path: str) -> bool:
        """Play audio file (cross-platform)"""
        try:
            if self.platform == "Darwin":  # macOS
                subprocess.run(['afplay', audio_path], check=True)
            elif self.platform == "Windows":
                # Use Windows Media Player
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'], check=True)
            elif self.platform == "Linux":
                # Try different Linux audio players
                for player in ['paplay', 'aplay', 'ffplay']:
                    try:
                        subprocess.run([player, audio_path], check=True, timeout=60)
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                return False
            return True
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
    
    def _speak_with_say(self, text: str) -> bool:
        """Fallback to macOS 'say' command"""
        try:
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            subprocess.run(['say', text], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Say command failed: {e}")
            return False
    
    def _speak_with_pyttsx3(self, text: str) -> bool:
        """Use pyttsx3 for Windows SAPI"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Limit text length
            if len(text) > 5000:
                text = text[:5000] + "..."
            
            engine.say(text)
            engine.runAndWait()
            return True
        except ImportError:
            print("pyttsx3 not installed. Install with: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"pyttsx3 error: {e}")
            return False
    
    def _speak_with_espeak(self, text: str) -> bool:
        """Use espeak for Linux"""
        try:
            # Limit text length
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            subprocess.run(['espeak', text], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"espeak command failed: {e}")
            return False
    
    def test_tts(self):
        """Test TTS functionality"""
        test_text = "Hello, this is a test of the text-to-speech system."
        print(f"\n🔊 Testing TTS on {self.platform}")
        print(f"Test text: {test_text}")
        print(f"Using method: {self.tts_method}")
        
        success = self.speak(test_text)
        if success:
            print("✅ TTS test successful")
        else:
            print("❌ TTS test failed")
        
        return success


if __name__ == "__main__":
    tts = TTSClient()
    tts.test_tts()

