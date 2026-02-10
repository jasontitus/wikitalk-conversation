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

# ============================================================
# Embedding model selection
# ============================================================
# Choose the embedding model based on your needs and hardware.
# IMPORTANT: Changing embedding model requires rebuilding the
# FAISS index (re-embedding all chunks). The index is model-specific.
#
# ORIGINAL OPTIONS:
#   all-MiniLM-L6-v2        384d   ~1,500-2,000 chunks/sec   ~4-5 hrs
#   all-mpnet-base-v2       768d   ~800-1,000 chunks/sec     ~9-12 hrs
#   BAAI/bge-m3             1024d  ~80-100 chunks/sec        ~100-120 hrs
#
# NEW OPTIONS (2025-2026 upgrades):
#   intfloat/e5-small-v2    384d   Same speed as MiniLM, much better retrieval
#                                  Requires "query: " / "passage: " prefixes (handled automatically)
#   BAAI/bge-small-en-v1.5  384d   MTEB ~62 vs MiniLM's ~56, similar speed
#   nomic-ai/modernbert-embed-base  768d  Flash Attention 2, 8192 token context

# ===== SELECT YOUR MODEL HERE =====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FAST (default)
# EMBEDDING_MODEL = "intfloat/e5-small-v2"                  # UPGRADED: same 384d, much better retrieval
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"                # UPGRADED: same 384d, MTEB ~62
# EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"        # UPGRADED: 768d, modern architecture
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # BALANCED
# EMBEDDING_MODEL = "BAAI/bge-m3"                           # HIGH QUALITY (slow but best)

# Models that require query/passage prefixes for best results
_EMBEDDING_PREFIX_MAP = {
    "e5": {"query": "query: ", "passage": "passage: "},
    "bge": {"query": "Represent this sentence for searching relevant passages: ", "passage": ""},
    "nomic": {"query": "search_query: ", "passage": "search_document: "},
    "modernbert-embed": {"query": "search_query: ", "passage": "search_document: "},
}

def get_embedding_prefixes(model_name: str) -> dict:
    """Get query/passage prefixes for the current embedding model."""
    for key, prefixes in _EMBEDDING_PREFIX_MAP.items():
        if key in model_name.lower():
            return prefixes
    return {"query": "", "passage": ""}

EMBEDDING_PREFIXES = get_embedding_prefixes(EMBEDDING_MODEL)

# Set embedding dimension based on model
if "bge-m3" in EMBEDDING_MODEL:
    EMBEDDING_DIM = 1024
elif any(x in EMBEDDING_MODEL for x in ["mpnet-base", "modernbert-embed", "nomic-embed"]):
    EMBEDDING_DIM = 768
else:
    EMBEDDING_DIM = 384

# ============================================================
# Retrieval configuration
# ============================================================
RETRIEVAL_TOPK = 40
MEMORY_TURNS = 8
TEMPERATURE = 0.2

# FAISS Index type selection
# Options: "flat" (original), "ivfpq", "ivfsq", "hnsw"
FAISS_INDEX_TYPE = "ivfpq"

# ============================================================
# LLM configuration
# ============================================================
# Uses OpenAI-compatible API (LM Studio, llama.cpp, etc.)
#
# ORIGINAL:
#   Qwen2.5-14B-Instruct     14B params, ~20-25 tok/s
#
# NEW OPTIONS (2025-2026 upgrades):
#   Qwen3-8B                 Matches Qwen2.5-14B quality at ~2x speed (43% fewer params)
#                             Use non-thinking mode for fast Q&A. Trained on 36T tokens.
#   Qwen3-4B                 Maximum speed option. Good for query rewrite calls.
#   Qwen3-30B-A3B            MoE: 30B total but only 3.3B active. Needs 24GB+ VRAM.
#   Gemma-3-12B              Strong on FACTS Grounding benchmark. QAT quantization.
#   Llama-3.1-8B-Instruct    Most factually conservative at 8B size.

LLM_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "Qwen2.5-14B-Instruct"
# LLM_MODEL = "Qwen3-8B"                    # RECOMMENDED UPGRADE: same quality, ~2x faster
# LLM_MODEL = "Qwen3-4B"                    # MAX SPEED: good enough for grounded Q&A
# LLM_MODEL = "Qwen3-30B-A3B"               # BEST QUALITY: MoE, needs 24GB+ VRAM
# LLM_MODEL = "Gemma-3-12B"                 # STRONG: best FACTS Grounding scores
# LLM_MODEL = "Llama-3.1-8B-Instruct"       # CONSERVATIVE: most factually grounded at 8B

# Query rewriting adds an extra LLM call per query to make follow-up
# questions more specific. Adds 0.5-2s latency per turn.
# Disable for snappier conversation at the cost of worse follow-up handling.
QUERY_REWRITE_ENABLED = True
# QUERY_REWRITE_ENABLED = False              # FASTER: save 0.5-2s per turn

# ============================================================
# TTS configuration
# ============================================================
# TTS_ENGINE selects which text-to-speech backend to use.
# Options: "auto", "kokoro", "piper", "say", "espeak", "pyttsx3"
#   "auto"   - Try kokoro first, then piper, then platform fallback (recommended)
#   "kokoro" - Kokoro TTS: 82M params, natural-sounding, 54 voices (pip install kokoro-onnx)
#   "piper"  - Piper TTS: lightweight ONNX, fast but robotic at low quality
#   "say"    - macOS built-in
#   "espeak" - Linux built-in
#   "pyttsx3"- Windows SAPI

TTS_ENGINE = "auto"
# TTS_ENGINE = "kokoro"                      # RECOMMENDED UPGRADE: dramatically better voice quality

# Kokoro TTS settings (used when TTS_ENGINE is "kokoro" or "auto")
# Voices: af_heart, af_bella, af_nova, af_sky, am_adam, am_michael, bf_emma, bm_george, ...
# Full list: https://huggingface.co/hexgrad/Kokoro-82M
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0

# Piper TTS settings (used when TTS_ENGINE is "piper" or "auto" fallback)
HOME_DIR = Path.home()
SYSTEM = platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'

PIPER_VOICE_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx"
PIPER_CONFIG_PATH = HOME_DIR / "piper_voices" / "en_US-amy-low.onnx.json"

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

# ============================================================
# STT (Speech-to-Text) configuration
# ============================================================
# STT_ENGINE selects which speech-to-text backend to use for voice input.
# Options: "none", "auto", "moonshine", "faster_whisper"
#   "none"           - Text input only via CLI (original behavior)
#   "auto"           - Try moonshine first, then faster_whisper, then fall back to text
#   "moonshine"      - Moonshine: 61M params, variable-length (no 30s padding), fast (pip install moonshine)
#   "faster_whisper"  - Faster-Whisper: Whisper accuracy, 4x faster via CTranslate2 (pip install faster-whisper)

STT_ENGINE = "none"
# STT_ENGINE = "auto"                        # RECOMMENDED: adds voice input
# STT_ENGINE = "moonshine"                   # BEST LATENCY: purpose-built for conversation
# STT_ENGINE = "faster_whisper"              # BEST ACCURACY: Whisper-level quality

# Moonshine settings
MOONSHINE_MODEL = "moonshine/base"           # Options: "moonshine/tiny" (27M), "moonshine/base" (61M)

# Faster-Whisper settings
FASTER_WHISPER_MODEL = "base.en"             # Options: "tiny.en", "base.en", "small.en", "medium.en"
FASTER_WHISPER_DEVICE = "auto"               # "auto", "cpu", "cuda"

# Silence detection: seconds of silence before processing speech
STT_SILENCE_DURATION = 1.5

# ============================================================
# Text processing
# ============================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Processing configuration - optimized for 128GB RAM
MAX_WORKERS = 2
BATCH_SIZE = 2000

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)

