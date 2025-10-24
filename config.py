"""
WikiTalk Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FINEWIKI_DIR = BASE_DIR / "finewiki" / "data" / "enwiki"

# Data storage paths
SQLITE_DB_PATH = DATA_DIR / "docs.sqlite"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
IDS_MAPPING_PATH = DATA_DIR / "ids.bin"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # BGE-M3 embedding dimension

# Retrieval configuration
RETRIEVAL_TOPK = 40
MEMORY_TURNS = 8
TEMPERATURE = 0.2

# LLM configuration
LLM_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "Qwen2.5-14B-Instruct"

# TTS configuration - point to home directory voices and piper executable
HOME_DIR = Path.home()
PIPER_VOICE_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx"
PIPER_CONFIG_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = HOME_DIR / "experiments" / "piper" / "build" / "piper"

# Add Piper to PATH if it exists
if PIPER_EXECUTABLE.exists():
    piper_dir = str(PIPER_EXECUTABLE.parent)
    if piper_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{piper_dir}:{os.environ.get('PATH', '')}"

# Text processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Processing configuration - optimized for 128GB RAM
MAX_WORKERS = 2  # Process 2 files at a time to avoid memory issues
BATCH_SIZE = 2000  # Larger batch size for better performance

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)

