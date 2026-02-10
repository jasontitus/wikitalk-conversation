"""
Speech-to-Text client for WikiTalk
Backends: Moonshine (recommended), Faster-Whisper, or text-only fallback

Usage:
    pip install moonshine           # For Moonshine backend
    pip install faster-whisper      # For Faster-Whisper backend

Both backends require a microphone. Audio capture uses sounddevice (pip install sounddevice).
"""
import logging
import numpy as np
from typing import Optional

from config import (
    STT_ENGINE, MOONSHINE_MODEL, FASTER_WHISPER_MODEL,
    FASTER_WHISPER_DEVICE, STT_SILENCE_DURATION,
)

logger = logging.getLogger(__name__)

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1


class STTClient:
    def __init__(self):
        self.engine = self._select_engine()
        self._moonshine_model = None
        self._whisper_model = None

    def _select_engine(self) -> str:
        """Select the best STT engine based on config."""
        engine = STT_ENGINE.lower()

        if engine == "none":
            return "none"

        if engine == "moonshine":
            if self._check_moonshine_available():
                return "moonshine"
            logger.warning("Moonshine requested but not available. Install: pip install moonshine sounddevice")
            return "none"

        if engine == "faster_whisper":
            if self._check_faster_whisper_available():
                return "faster_whisper"
            logger.warning("Faster-Whisper requested but not available. Install: pip install faster-whisper sounddevice")
            return "none"

        # Auto mode
        if self._check_moonshine_available():
            return "moonshine"
        if self._check_faster_whisper_available():
            return "faster_whisper"

        return "none"

    @staticmethod
    def _check_moonshine_available() -> bool:
        try:
            import moonshine  # noqa: F401
            import sounddevice  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_faster_whisper_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            import sounddevice  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        return self.engine != "none"

    def listen(self) -> Optional[str]:
        """Record from microphone until silence, then transcribe.

        Returns the transcribed text, or None on failure.
        """
        if self.engine == "none":
            return None

        try:
            audio = self._record_until_silence()
            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                return None

            if self.engine == "moonshine":
                return self._transcribe_moonshine(audio)
            elif self.engine == "faster_whisper":
                return self._transcribe_faster_whisper(audio)
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None

    def _record_until_silence(self) -> Optional[np.ndarray]:
        """Record audio from microphone until silence is detected."""
        import sounddevice as sd

        chunk_duration = 0.5  # seconds per chunk
        chunk_samples = int(SAMPLE_RATE * chunk_duration)
        silence_threshold = 0.01
        silence_chunks_needed = int(STT_SILENCE_DURATION / chunk_duration)

        chunks = []
        silent_chunks = 0
        has_speech = False

        print("(listening...)", end=" ", flush=True)

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32') as stream:
                while True:
                    data, _ = stream.read(chunk_samples)
                    chunk = data.flatten()
                    rms = np.sqrt(np.mean(chunk ** 2))

                    if rms > silence_threshold:
                        has_speech = True
                        silent_chunks = 0
                        chunks.append(chunk)
                    else:
                        if has_speech:
                            chunks.append(chunk)
                            silent_chunks += 1
                            if silent_chunks >= silence_chunks_needed:
                                break
                        # If no speech detected yet, keep waiting
        except KeyboardInterrupt:
            return None

        if not chunks:
            return None

        return np.concatenate(chunks)

    def _transcribe_moonshine(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using Moonshine."""
        import moonshine

        if self._moonshine_model is None:
            logger.info(f"Loading Moonshine model: {MOONSHINE_MODEL} (first use)...")
            self._moonshine_model = moonshine.load_model(MOONSHINE_MODEL)
            logger.info("Moonshine model loaded.")

        # Moonshine expects (1, samples) float32 array
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        tokens = moonshine.transcribe(self._moonshine_model, audio)
        text = moonshine.decode(tokens)
        if isinstance(text, list):
            text = text[0]
        return text.strip() if text else None

    def _transcribe_faster_whisper(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using Faster-Whisper."""
        from faster_whisper import WhisperModel

        if self._whisper_model is None:
            device = FASTER_WHISPER_DEVICE
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            compute_type = "int8" if device == "cpu" else "float16"
            logger.info(f"Loading Faster-Whisper model: {FASTER_WHISPER_MODEL} on {device} (first use)...")
            self._whisper_model = WhisperModel(
                FASTER_WHISPER_MODEL, device=device, compute_type=compute_type
            )
            logger.info("Faster-Whisper model loaded.")

        segments, _ = self._whisper_model.transcribe(audio, beam_size=1, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        return text if text else None

    def test_stt(self) -> bool:
        """Test STT by recording a short phrase."""
        if not self.is_available:
            print(f"STT not available (engine={STT_ENGINE}).")
            print("Install one of:")
            print("  pip install moonshine sounddevice     # Moonshine (recommended)")
            print("  pip install faster-whisper sounddevice # Faster-Whisper")
            return False

        print(f"STT engine: {self.engine}")
        print("Say something (will stop after silence)...")
        text = self.listen()
        if text:
            print(f"Heard: \"{text}\"")
            return True
        else:
            print("No speech detected.")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = STTClient()
    client.test_stt()
