"""
Configuration for WikiTalk system
"""
import os
from pathlib import Path
import platform

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FINEWIKI_DIR = BASE_DIR / "finewiki" / "data" / "enwiki"

# Data storage paths
SQLITE_DB_PATH = DATA_DIR / "docs.sqlite"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
IDS_MAPPING_PATH = DATA_DIR / "ids.bin"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

# Embedding model selection
# ===========================
# Choose the embedding model based on your needs and hardware:
#
# RECOMMENDED (Fast, Good Quality):
#   all-MiniLM-L6-v2
#   - Speed: 1,500-2,000 chunks/sec on Mac GPU
#   - Build time: ~4-5 hours
#   - Quality: Excellent for semantic search
#   - Dimensions: 384
#   - Best for: Getting started, quick builds
#
# HIGH QUALITY (Slower, Best Results):
#   BAAI/bge-m3
#   - Speed: 80-100 chunks/sec on Mac GPU
#   - Build time: ~100-120 hours (one-time, can interrupt/resume)
#   - Quality: Highest quality embeddings
#   - Dimensions: 1024
#   - Best for: Final production when quality is critical
#
# BALANCED (Medium Speed/Quality):
#   all-mpnet-base-v2
#   - Speed: 800-1,000 chunks/sec on Mac GPU
#   - Build time: ~9-12 hours
#   - Quality: Very good
#   - Dimensions: 768
#   - Best for: Balance of speed and quality

# ===== SELECT YOUR MODEL HERE =====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ FAST (default)
# EMBEDDING_MODEL = "BAAI/bge-m3"  # HIGH QUALITY (slow but best)
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # BALANCED

# Set embedding dimension based on model
if "bge-m3" in EMBEDDING_MODEL:
    EMBEDDING_DIM = 1024
elif "mpnet-base" in EMBEDDING_MODEL:
    EMBEDDING_DIM = 768
else:
    EMBEDDING_DIM = 384

# Retrieval configuration
RETRIEVAL_TOPK = 40  # Restored from 10 for better search quality (FAISS speed not affected by this)
MEMORY_TURNS = 8
TEMPERATURE = 0.2

# FAISS Index type selection
# Options: "flat" (original), "ivfpq", "ivfsq", "hnsw"
FAISS_INDEX_TYPE = "ivfpq"  # Change to "ivfpq", "ivfsq", or "hnsw" to use optimized indexes

# LLM configuration
LLM_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "Qwen2.5-14B-Instruct"

# TTS configuration - cross-platform support
HOME_DIR = Path.home()
SYSTEM = platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'

# Piper voice files (same location on all platforms)
PIPER_VOICE_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx"
PIPER_CONFIG_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx.json"

# Piper executable path (platform-specific)
if SYSTEM == "Windows":
    PIPER_EXECUTABLE = HOME_DIR / "experiments" / "piper" / "build" / "piper.exe"
elif SYSTEM == "Darwin":  # macOS
    PIPER_EXECUTABLE = HOME_DIR / "experiments" / "piper" / "build" / "piper"
else:  # Linux
    PIPER_EXECUTABLE = HOME_DIR / "experiments" / "piper" / "build" / "piper"

# Add Piper to PATH if it exists
if PIPER_EXECUTABLE.exists():
    piper_dir = str(PIPER_EXECUTABLE.parent)
    if piper_dir not in os.environ.get('PATH', ''):
        if SYSTEM == "Windows":
            os.environ['PATH'] = f"{piper_dir};{os.environ.get('PATH', '')}"
        else:
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

