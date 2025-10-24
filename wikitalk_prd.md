# WikiTalk: Local Conversational Historian

## 1. Product Overview

### Product Name
**WikiTalk (Local Conversational Historian)**

### Summary
WikiTalk is an offline, conversational AI assistant that allows users to **talk about history, science, and culture** using a **local copy of Wikipedia**. It understands natural questions, supports **follow-up questions and contextual discussion**, and **speaks answers aloud** via Piper TTS.

All data processing, model inference, and speech synthesis happen **entirely locally** on macOS — no cloud API calls, ensuring privacy and offline functionality.

---

## 2. Core Value Proposition

- **Offline knowledge:** Runs entirely from a local Wikipedia dataset (FineWiki Parquet files).
- **Natural conversation:** Multi-turn dialogue with context retention and topic continuity.
- **Voice interaction:** Speaks answers using Piper TTS.
- **Grounded knowledge:** Uses retrieved, cited chunks from Wikipedia to reduce hallucinations.
- **Mac-native:** Optimized for Apple Silicon performance.

---

## 3. Target Users

| Segment | Description | Key Benefit |
|----------|--------------|--------------|
| History Enthusiasts | Users who explore historical events and cause/effect relationships | Can discuss complex historical threads naturally |
| Students & Researchers | Seek to understand topics interactively without internet | Offline, reliable, source-grounded responses |
| Privacy-Conscious Users | Avoid cloud-based LLMs | 100% offline processing |
| Makers & Tinkerers | Experiment with datasets and LLMs | Transparent, modifiable system |

---

## 4. User Stories

1. **Topic Inquiry:** Ask, "Why did World War I start?" → learn from local Wikipedia.
2. **Follow-up Question:** Ask, "And what was Germany’s role?" → bot remembers prior context.
3. **Cross-topic Transition:** Ask, "How was that like the Napoleonic wars?" → context preserved.
4. **Voice Output:** Bot speaks answer naturally through Piper.
5. **Citations:** Bot lists which Wikipedia sections were used.
6. **Offline Use:** Works without internet access.

---

## 5. Functional Requirements

### 5.1 Data Layer

**Source:** `HuggingFaceFW/finewiki` Parquet dataset.

**Storage and Indexing:**
- Parse Parquet to structured chunks (700–1200 tokens, 100–200 overlap).
- Store:
  - SQLite DB (`docs.sqlite`) with FTS5 for BM25 search.
  - FAISS vector index (`faiss.index`) for dense retrieval.
  - ID mapping file (`ids.bin`).

### 5.2 Embedding Engine
- Model: `BAAI/bge-m3`
- Cosine-normalized FAISS vectors
- Hybrid: BM25 (FTS5) + dense retrieval
- Ranking: RapidFuzz or `bge-reranker-large`

### 5.3 Retrieval and Reranking
- Retrieve top 40 from BM25 + 40 from FAISS.
- Merge unique hits.
- Score semantic similarity with RapidFuzz.
- Optionally rerank top 20 with cross-encoder reranker.

### 5.4 LLM Layer

**Interface:** OpenAI-compatible REST (LM Studio or llama.cpp server)

**Model Options:**
- `Qwen2.5-14B-Instruct` (GGUF Q4_K_M)
- `Llama-3.1-8B-Instruct`

**Prompt Template:**
```
System: You are a factual historian using provided Wikipedia sources.
Cite with [1], [2]. Ask clarifying questions when info is missing.

User: {query}

Sources:
[1] World War I / Causes
...
[2] Archduke Franz Ferdinand / Assassination
...

Answer:
```

**Conversation Memory:**
- Rolling buffer (6–10 exchanges)
- Query rewritten using history before retrieval
- Stored in JSON or memory

### 5.5 Dialogue Flow

1. User input (text or voice)
2. Query rewrite based on history
3. Wikipedia retrieval (BM25 + FAISS + rerank)
4. Context + query → LLM
5. LLM outputs grounded answer
6. Piper generates speech
7. Exchange stored in history

### 5.6 TTS Integration

**Engine:** Piper
```
./piper -m voices/en_US-amy-medium.onnx -c voices/en_US-amy-medium.onnx.json --output_file out.wav
afplay out.wav
```
**Fallback:** macOS `say`

### 5.7 Conversation Persistence

Stored in `data/conversations/session_{timestamp}.json`:
```json
{
  "history": [
    {"role": "user", "content": "Why did WWI start?"},
    {"role": "assistant", "content": "It began after the assassination ... [1]"}
  ],
  "last_topic": "World War I"
}
```

---

## 6. Non-Functional Requirements

| Category | Requirement |
|-----------|--------------|
| Performance | <1 s retrieval; <10 s total answer time |
| Footprint | <30 GB with full English Wikipedia |
| Privacy | Fully offline |
| Compatibility | macOS 13+ (Intel & Apple Silicon) |
| Resilience | Fallback if FAISS/Piper unavailable |
| Transparency | Include source titles |
| Extensibility | Multi-language / custom wikis |

---

## 7. Technical Architecture

```
 ┌──────────────────────────┐
 │       User (Text/Voice)  │
 └───────────┬──────────────┘
             │
             ▼
 ┌──────────────────────────┐
 │ Input Parser / Whisper   │
 └───────────┬──────────────┘
             ▼
 ┌──────────────────────────┐
 │ Query Rewriter (LLM)     │
 └───────────┬──────────────┘
             ▼
 ┌──────────────────────────┐
 │ Retriever                │
 │  - SQLite FTS5 (BM25)    │
 │  - FAISS Dense Index     │
 │  - RapidFuzz Reranker    │
 └───────────┬──────────────┘
             ▼
 ┌──────────────────────────┐
 │ Local LLM (LM Studio)    │
 │  - Qwen2.5-14B-Instruct  │
 │  - Context: top chunks   │
 └───────────┬──────────────┘
             ▼
 ┌──────────────────────────┐
 │ Response Composer        │
 │  - Add citations [1],[2] │
 │  - Append to history     │
 └───────────┬──────────────┘
             ▼
 ┌──────────────────────────┐
 │ Piper TTS Output         │
 └───────────┬──────────────┘
             ▼
        Speaker Output
```

---

## 8. Configuration Options

| Setting | Description | Default |
|----------|--------------|----------|
| `DATA_PATH` | Path to FineWiki Parquet files | `~/data/finewiki/` |
| `LLM_URL` | Local LLM endpoint | `http://localhost:1234/v1/chat/completions` |
| `VOICE_PATH` | Path to Piper model | `./voices/en_US-amy-medium.onnx` |
| `RETRIEVAL_TOPK` | Top passages per method | `40` |
| `MEMORY_TURNS` | # dialogue turns to retain | `8` |
| `TEMPERATURE` | LLM generation temperature | `0.2` |

---

## 9. Future Enhancements

1. **Interactive GUI:** SwiftUI/Electron front-end with mic input.
2. **Language Switching:** Multi-lingual FineWiki support.
3. **Personal Notes Layer:** Include user-added wiki snippets.
4. **Whisper Integration:** Real-time voice input.
5. **Dataset Updates:** Refresh Wikipedia incrementally.

---

## 10. Example Flow

**User:** “Tell me about the Meiji Restoration.”  
**Bot:** “The Meiji Restoration was Japan’s 1868 political revolution restoring imperial rule [1]. It led to modernization and industrialization [2].” *(spoken via Piper)*

**User:** “And how did it affect Korea?”  
**Bot:** “Japan’s reforms strengthened its military, enabling control over Korea in the 1890s [3].” *(context maintained)*

