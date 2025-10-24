# 🧠 Embedding Search for WikiTalk

## Overview

The embedding search feature enables **semantic search** on the entire Wikipedia dataset (33.5M chunks). Unlike keyword-based search, embedding search understands the *meaning* of your query, making it perfect for conceptual questions like:

- "Tell me about ancient roman architecture"
- "What are quantum computing applications?"
- "How did renaissance art develop?"

## How It Works

### 1. Embedding Generation

Each chunk is converted to a dense vector (embedding) that captures its semantic meaning:

```
Text: "The Roman Forum was the center of ancient Roman politics and justice..."
       ↓
Embedding: [0.12, -0.45, 0.89, ..., 0.23] (384 dimensions)
```

### 2. FAISS Index

All embeddings are stored in a FAISS (Facebook AI Similarity Search) index for fast similarity search:

```
Query: "ancient roman architecture"
   ↓
Query Embedding: [0.10, -0.47, 0.91, ..., 0.21]
   ↓
FAISS: Find similar embeddings
   ↓
Results: Roman Forum, Colosseum, Pantheon, Aqueducts, ...
```

### 3. Reranking

Results are reranked using fuzzy matching for final relevance:

```
Top 10 from FAISS
   ↓
Fuzzy match against query
   ↓
Return top 3-5 most relevant
```

## Building the Index

### One-Time Setup

```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate

# Build the embedding index (takes 1-2 hours on GPU, 4-6 hours on CPU)
time python build_embeddings.py
```

The script will:
1. Confirm you want to proceed
2. Show estimated build time based on your hardware
3. Stream chunks from database and generate embeddings
4. Create and save FAISS index (~15GB)
5. Display completion statistics

### Progress Tracking

During build, you'll see:

```
🚀 Building embedding index from database (streaming)...
   This will take 1-2 hours for 33.5M chunks
   Total chunks to process: 33,477,070
   ✓ Processed 50,000/33,477,070 chunks
   ✓ Processed 100,000/33,477,070 chunks
   ...
   ✓ Processed 33,477,070/33,477,070 chunks
✅ Embedding index created: 33,477,070 vectors
```

## Using Embedding Search

### In WikiTalk

Once the index is built, WikiTalk automatically uses semantic search:

```bash
python wikitalk.py

User: Tell me about ancient roman architecture
WikiTalk: (searches using embeddings)
         Loading 5 relevant articles from Wikipedia...
         [Articles about Roman Forum, Colosseum, etc.]
         
         Roman architecture, spanning over 1,000 years,
         represents one of history's most influential...
```

### Programmatically

```python
from retriever import HybridRetriever

# Initialize retriever (loads existing index)
retriever = HybridRetriever(use_bm25_only=False)
retriever.load_indexes()

# Semantic search
results = retriever.search(
    query="ancient roman architecture",
    top_k=5,
    method="embedding"  # Use embedding search
)

for result in results:
    print(f"{result['title']}: {result['text'][:200]}...")

retriever.close()
```

## Search Methods

### Embedding Search (Semantic)

**Best for**: Conceptual queries, descriptions, topics

```python
results = retriever.search(query, method="embedding")
```

**Advantages**:
- Understands meaning and context
- Great for multi-word phrases
- Finds semantically related content
- Works across topics

**Disadvantages**:
- Requires index (1-2 hours to build)
- Slower first query (needs to generate embedding)
- Uses more memory (~15GB)

**Example queries**:
- "ancient roman architecture"
- "quantum physics breakthroughs"
- "renaissance art movements"
- "climate change effects"

### LIKE Search (Keyword)

**Best for**: Specific terms, fallback when embedding fails

```python
results = retriever.search(query, method="like")
```

**Advantages**:
- No build time needed
- Fast and lightweight
- Works immediately
- Good for exact matches

**Disadvantages**:
- Keyword-based only
- Doesn't understand meaning
- Struggles with multi-word concepts

**Example queries**:
- "Shakespeare"
- "World War I"
- "Python programming"

## Performance

### Index Building

| Hardware | Time | Cost |
|----------|------|------|
| NVIDIA GPU (RTX 4090) | 1-2 hours | Fastest |
| NVIDIA GPU (RTX 4070) | 2-3 hours | Good |
| Apple Silicon (M3 Max) | 4-6 hours | Acceptable |
| CPU (i7-13700K) | 4-6 hours | Acceptable |
| Mac CPU | 6-12 hours | Slow but works |

### Search Performance

| Operation | Time |
|-----------|------|
| Query embedding generation | 0.02-0.05s |
| FAISS similarity search | 0.05-0.10s |
| Result reranking | 0.01-0.05s |
| **Total per query** | **0.1-0.2s** |

### Storage

- FAISS index: ~15GB
- ID mapping: ~200MB
- **Total**: ~15.2GB

## Architecture

```
WikiTalk Embedding Search
│
├─ Dense Vectors (33.5M chunks)
│  └─ Stored in FAISS index
│
├─ Query Processing
│  ├─ Generate query embedding
│  ├─ Search FAISS index
│  └─ Fetch chunk metadata from SQLite
│
└─ Result Processing
   ├─ Rerank using fuzzy matching
   ├─ Format for display
   └─ Return top results
```

## Troubleshooting

### Issue: "Embedding search not available"

**Cause**: Index hasn't been built yet

**Fix**:
```bash
python build_embeddings.py
```

### Issue: Build process hangs or crashes

**Cause**: Out of memory or GPU issues

**Fix**:
```bash
# Restart and try again
# Reduce batch sizes in retriever.py if needed:
# embedding_batch_size = 256  (instead of 512)
# db_batch_size = 2500        (instead of 5000)
```

### Issue: Slow search results

**Cause**: FAISS queries taking time, or CPU at 100%

**Fix**: 
- Wait for index to stabilize (first few queries are slower)
- Close other applications to free memory
- If still slow, switch to LIKE search as fallback

## Advanced Configuration

### Adjust Embedding Batch Size

Edit `retriever.py`:

```python
def _build_embedding_index_streaming(self):
    # For lower memory systems:
    embedding_batch_size = 256   # Default: 512
    db_batch_size = 2500          # Default: 5000
```

### Use Different Embedding Model

Edit `config.py`:

```python
# Current model (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Larger model (more accurate, slower):
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Smaller model (faster, less accurate):
# EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
```

## Monitoring Build Progress

The build process logs detailed information:

```bash
# Watch progress in real-time
tail -f build_embeddings.log

# Or redirect when running
python build_embeddings.py | tee build_embeddings.log
```

## Integration with LLM

When using embeddings with the LLM:

```
User Query
   ↓
Embedding Search (finds 5 relevant articles)
   ↓
LLM generates response using articles as context
   ↓
Response with citations
```

Example flow:

```
User: "Tell me about ancient roman architecture"

1. Embedding Search finds:
   - Roman Forum article
   - Colosseum article
   - Roman engineering article
   - Pantheon article
   - Aqueduct systems article

2. LLM uses these as context to generate comprehensive answer

3. Response: "Roman architecture combined engineering 
   brilliance with aesthetic design. The Colosseum 
   demonstrates their mastery of large-scale structures [1],
   while the Pantheon showcases innovative concrete 
   construction [2]..."
```

## Comparison: Embedding vs LIKE Search

| Feature | Embedding | LIKE |
|---------|-----------|------|
| **Search Type** | Semantic | Keyword |
| **Build Time** | 1-2 hours | None |
| **Query Speed** | 0.1-0.2s | 0.5-3s |
| **Memory** | 15GB index | Minimal |
| **Quality** | Excellent for concepts | Good for exact terms |
| **Best For** | Descriptive queries | Specific terms |

## Next Steps

1. **Build the index**:
   ```bash
   python build_embeddings.py
   ```

2. **Test semantic search**:
   ```bash
   python -c "
   from retriever import HybridRetriever
   r = HybridRetriever()
   r.load_indexes()
   results = r.search('ancient roman architecture', method='embedding', top_k=3)
   for res in results:
       print(f'{res[\"title\"]}: {res[\"text\"][:100]}...')
   r.close()
   "
   ```

3. **Use in WikiTalk**:
   ```bash
   python wikitalk.py
   ```

## See Also

- [[FINAL_STATUS.md]] - System status and capabilities
- [[config.py]] - Configuration options
- [[retriever.py]] - Search implementation details
- [[llm_client.py]] - LLM integration
