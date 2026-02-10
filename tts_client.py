"""
Text-to-Speech client for WikiTalk
Cross-platform support: Windows, macOS, Linux
Backends: Kokoro (recommended), Piper, platform fallbacks
"""
import subprocess
import os
import tempfile
import platform
import logging
from pathlib import Path
from typing import Optional

from config import *

logger = logging.getLogger(__name__)


class TTSClient:
    def __init__(self):
        self.platform = platform.system()
        self.piper_voice_path = PIPER_VOICE_PATH
        self.piper_config_path = PIPER_CONFIG_PATH
        self._kokoro_model = None
        self._kokoro_voicepack = None
        self.tts_method = self._select_tts_method()

    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for TTS by removing markdown and Wikipedia formatting"""
        import re

        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'—|–|‑', '-', text)
        text = re.sub(r'•', ' ', text)
        text = re.sub(r'\n\n+', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = ' '.join(text.split())

        return text

    def _select_tts_method(self) -> str:
        """Select the best TTS method based on TTS_ENGINE config"""
        engine = TTS_ENGINE.lower()

        if engine == "kokoro":
            if self._check_kokoro_availability():
                return "kokoro"
            logger.warning("Kokoro TTS requested but not available. Install: pip install kokoro-onnx")
            return "silent"

        if engine == "piper":
            if self._check_piper_availability():
                return "piper"
            logger.warning("Piper TTS requested but not available.")
            return "silent"

        if engine == "say":
            return "say" if self.platform == "Darwin" else "silent"

        if engine == "espeak":
            return "espeak" if self._check_espeak_availability() else "silent"

        if engine == "pyttsx3":
            return "pyttsx3" if self._check_pyttsx3_availability() else "silent"

        # Auto mode: try best options in order
        if self._check_kokoro_availability():
            return "kokoro"

        if self._check_piper_availability():
            return "piper"

        if self.platform == "Darwin":
            return "say"
        elif self.platform == "Windows":
            if self._check_pyttsx3_availability():
                return "pyttsx3"
        elif self.platform == "Linux":
            if self._check_espeak_availability():
                return "espeak"

        return "silent"

    def _check_kokoro_availability(self) -> bool:
        """Check if Kokoro TTS (kokoro-onnx) is available"""
        try:
            import kokoro_onnx  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_piper_availability(self) -> bool:
        """Check if Piper TTS is available"""
        try:
            if not os.path.exists(self.piper_voice_path):
                return False
            if not os.path.exists(self.piper_config_path):
                return False
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
            import pyttsx3  # noqa: F401
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

        text = self._clean_text_for_speech(text)

        try:
            if self.tts_method == "kokoro":
                return self._speak_with_kokoro(text)
            elif self.tts_method == "piper":
                return self._speak_with_piper(text)
            elif self.tts_method == "say":
                return self._speak_with_say(text)
            elif self.tts_method == "pyttsx3":
                return self._speak_with_pyttsx3(text)
            elif self.tts_method == "espeak":
                return self._speak_with_espeak(text)
            else:
                logger.warning("No TTS method available")
                return False
        except Exception as e:
            print(f"TTS error: {e}")
            return False

    def _speak_with_kokoro(self, text: str) -> bool:
        """Use Kokoro TTS for high-quality speech synthesis"""
        try:
            import kokoro_onnx
            import soundfile as sf

            # Lazy-load the model on first use
            if self._kokoro_model is None:
                logger.info("Loading Kokoro TTS model (first use)...")
                self._kokoro_model = kokoro_onnx.Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
                logger.info(f"Kokoro TTS loaded, voice: {KOKORO_VOICE}")

            # Generate audio
            samples, sample_rate = self._kokoro_model.create(
                text, voice=KOKORO_VOICE, speed=KOKORO_SPEED
            )

            # Write to temp file and play
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, samples, sample_rate)

            self._play_audio(temp_path)
            os.unlink(temp_path)
            return True

        except ImportError as e:
            logger.error(f"Kokoro dependency missing: {e}")
            logger.error("Install with: pip install kokoro-onnx soundfile")
            logger.error("Download models: kokoro-v1.0.onnx and voices-v1.0.bin from HuggingFace hexgrad/Kokoro-82M")
            return False
        except FileNotFoundError:
            logger.error("Kokoro model files not found.")
            logger.error("Download kokoro-v1.0.onnx and voices-v1.0.bin to the project directory.")
            logger.error("See: https://huggingface.co/hexgrad/Kokoro-82M")
            return False
        except Exception as e:
            logger.error(f"Kokoro TTS error: {e}")
            return False

    def _speak_with_piper(self, text: str) -> bool:
        """Use Piper TTS for speech synthesis (cross-platform)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            cmd = [
                'piper',
                '-m', str(self.piper_voice_path),
                '-c', str(self.piper_config_path),
                '--output_file', temp_path
            ]

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

            if os.path.exists(temp_path):
                self._play_audio(temp_path)
                os.unlink(temp_path)
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
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'], check=True)
            elif self.platform == "Linux":
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
        print(f"\nTesting TTS on {self.platform}")
        print(f"Test text: {test_text}")
        print(f"Using method: {self.tts_method}")

        success = self.speak(test_text)
        if success:
            print("TTS test successful")
        else:
            print("TTS test failed")

        return success


if __name__ == "__main__":
    tts = TTSClient()
    tts.test_tts()
