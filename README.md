# WikiTalk: Local Conversational Historian

<IN DEVELOPMENT - NOT CURRENTLY WORKING>

WikiTalk is an offline, conversational AI assistant that allows users to talk about history, science, and culture using a local copy of Wikipedia. It understands natural questions, supports follow-up questions and contextual discussion, and speaks answers aloud via Piper TTS.

## Features

- **Offline knowledge:** Runs entirely from a local Wikipedia dataset (FineWiki Parquet files)
- **Natural conversation:** Multi-turn dialogue with context retention and topic continuity
- **Voice interaction:** Speaks answers using Piper TTS
- **Grounded knowledge:** Uses retrieved, cited chunks from Wikipedia to reduce hallucinations
- **Mac-native:** Optimized for Apple Silicon performance

## Quick Start

### 1. Setup

```bash
# Install dependencies
python setup.py

# Or manually install requirements
pip install -r requirements.txt
```

### 2. Process Wikipedia Data

```bash
# This will create SQLite and FAISS indexes from the parquet files
python data_processor.py
```

**Note:** This step can take several hours depending on your system. The process will:
- Parse all parquet files in `finewiki/data/enwiki/`
- Create text chunks with overlap
- Build SQLite FTS5 index for BM25 search
- Generate embeddings and create FAISS index for dense retrieval

### 3. Start LLM Server

You need a local LLM server running. Options:

**Option A: LM Studio**
1. Download LM Studio from https://lmstudio.ai/
2. Load a model like `Qwen2.5-14B-Instruct` or `Llama-3.1-8B-Instruct`
3. Start the local server (usually on port 1234)

**Option B: llama.cpp**
```bash
# Download and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a model (example with Qwen2.5-14B)
# Start server
./server -m models/qwen2.5-14b-instruct.gguf --port 1234
```

### 4. Optional: Setup TTS

For voice output, download Piper TTS voices:

```bash
# Create voices directory
mkdir -p voices

# Download a voice (example)
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx -O voices/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json -O voices/en_US-amy-medium.onnx.json
```

### 5. Run WikiTalk

```bash
python wikitalk.py
```

## Usage

Once running, you can:

- Ask questions: "Tell me about the Meiji Restoration"
- Follow up: "And how did it affect Korea?"
- Get voice responses (if TTS is configured)
- Type `clear` to start a new conversation
- Type `quit` to exit

## Architecture

```
User Input → Query Rewriter → Hybrid Retrieval → LLM → Response → TTS → Audio
                ↓
        Conversation Memory ← SQLite + FAISS Indexes
```

### Components

- **Data Processor**: Parses parquet files and creates search indexes
- **Hybrid Retriever**: Combines BM25 (SQLite FTS5) and dense (FAISS) search
- **LLM Client**: Interfaces with local LLM for response generation
- **TTS Client**: Converts text to speech using Piper or macOS `say`
- **Conversation Manager**: Handles multi-turn dialogue context

## Configuration

Edit `config.py` to customize:

- Data paths and model settings
- Retrieval parameters (top-k, chunk size)
- LLM server URL and model
- TTS voice settings

## Requirements

- Python 3.8+
- 8GB+ RAM (for embeddings and FAISS index)
- 30GB+ disk space (for full English Wikipedia)
- Local LLM server (LM Studio or llama.cpp)
- Optional: Piper TTS for voice output

## Troubleshooting

### Data Processing Issues
- Ensure parquet files are in `finewiki/data/enwiki/`
- Check available disk space (30GB+ needed)
- Monitor memory usage during processing

### LLM Connection Issues
- Verify LLM server is running on correct port
- Check `LLM_URL` in config.py
- Test with: `curl http://localhost:1234/v1/models`

### TTS Issues
- Check Piper installation: `which piper`
- Verify voice files in `voices/` directory
- Falls back to macOS `say` command if Piper unavailable

## Performance

- **Retrieval**: <1 second for most queries
- **Total response time**: <10 seconds
- **Memory usage**: ~8GB for full English Wikipedia
- **Storage**: ~30GB for complete dataset

## License

This project uses Wikipedia data under CC BY-SA 4.0 license.

