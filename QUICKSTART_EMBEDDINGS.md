# 🚀 Quick Start: Embedding Search for WikiTalk

## What's Embedding Search?

Embedding search (also called semantic search) understands the **meaning** of your queries, not just keywords. Examples:

```
Query: "Tell me about ancient roman architecture"
↓
Embedding search finds: Roman Forum, Colosseum, Pantheon, Aqueducts
(understands you want architectural topics, not just "roman" keyword matches)
```

## Step 1: Check Your System

```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate

# Verify database exists
ls -lh data/docs.sqlite
# Should show: 81.0 GB

# Verify GPU availability (optional but recommended)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Step 2: Build Embedding Index (One-Time, 1-2 hours)

```bash
time python build_embeddings.py
```

**What happens:**
1. You'll see a confirmation prompt
2. Streaming starts - chunks loaded from database
3. Embeddings generated in batches
4. FAISS index created and saved (~15GB)

**Expected output:**
```
🚀 Building embedding index from database (streaming)...
   This will take 1-2 hours for 33.5M chunks
   Total chunks to process: 33,477,070
   ✓ Processed 50,000/33,477,070 chunks
   ✓ Processed 100,000/33,477,070 chunks
   ...
✅ Embedding index created: 33,477,070 vectors
```

**Estimated times:**
- GPU (RTX 4090): **1-2 hours** ⚡ Fastest
- GPU (RTX 4070): **2-3 hours**
- Apple Silicon (M3 Max): **4-6 hours**
- CPU: **6-12 hours**

**If interrupted:** Just run again, it will pick up where it left off.

## Step 3: Start LM Studio (Required for AI responses)

1. Open LM Studio app
2. Load a model (e.g., "Mistral" or "Neural Chat")
3. Click "Start Server"
4. Wait for "Server started at http://localhost:1234"

(Skip this if you only want search results without AI responses)

## Step 4: Use WikiTalk with Embeddings

```bash
python wikitalk.py
```

Example conversation:

```
WikiTalk initialized successfully!

User: Tell me about ancient roman architecture

Original query: Tell me about ancient roman architecture
Rewritten query: ancient roman architecture styles history
Searching Wikipedia...
Generating response...
Speaking response...

Response: Roman architecture represents one of history's most 
influential building traditions. The Romans pioneered the use of 
concrete, which allowed them to create vast domes and arches...

[Sources: Colosseum, Roman engineering, Ancient Rome, ...]
```

## Step 5: Test Just the Search (No LLM needed)

```bash
python -c "
from retriever import HybridRetriever
import time

r = HybridRetriever()  # Loads embedding index automatically
r.load_indexes()

# Test embedding search
start = time.time()
results = r.search('ancient roman architecture', top_k=5, method='embedding')
elapsed = time.time() - start

print(f'✓ Found {len(results)} results in {elapsed:.2f}s')
print()
for i, res in enumerate(results, 1):
    print(f'{i}. {res[\"title\"]}')
    print(f'   Score: {res.get(\"rerank_score\", 0):.3f}')
    print(f'   {res[\"text\"][:150]}...')
    print()

r.close()
"
```

## Troubleshooting

### Issue: "Embedding search not available"

**Solution:**
```bash
# Build the index
python build_embeddings.py
```

### Issue: Build takes too long or uses too much memory

**Solution - Reduce batch size:**

Edit `retriever.py` line ~97:
```python
embedding_batch_size = 256   # Change from 512
db_batch_size = 2500          # Change from 5000
```

Then rebuild:
```bash
python build_embeddings.py
```

### Issue: GPU out of memory during build

**Solution:**

Edit `retriever.py` line ~97:
```python
embedding_batch_size = 128   # Reduce batch size
```

Then retry build.

### Issue: Search is slow

**Solution:**

- First query is slower (model loading)
- Subsequent queries should be 0.1-0.2s
- If still slow, check disk I/O or memory pressure

## Comparing Search Methods

### Embedding Search (What you just built)

```python
results = retriever.search(query, method="embedding")
```

**Best for:**
- "Tell me about ancient roman architecture"
- "What are quantum computing applications?"
- "How did renaissance art develop?"
- Conceptual questions
- Multi-word phrases

**Speed:** 0.1-0.2 seconds per query

### Keyword Search (Fallback)

```python
results = retriever.search(query, method="like")
```

**Best for:**
- "Shakespeare"
- "World War I"
- "Python programming"
- Single word lookups
- Exact term matches

**Speed:** 0.5-3 seconds per query

## What's Happening Under The Hood

```
Your Query
    ↓
Sentence-Transformers model (384 dimensions)
    ↓
Query Embedding
    ↓
FAISS Index (33.5M vectors)
    ↓
Similarity Search (Cosine distance)
    ↓
Top 10 Results
    ↓
Fuzzy Reranking
    ↓
Top 3-5 Results to You
```

## Next Steps

1. **Try different queries:**
   ```bash
   python wikitalk.py
   # Try: "Tell me about quantum physics"
   # Try: "How does photosynthesis work?"
   # Try: "What is the history of aviation?"
   ```

2. **Add to your workflow:**
   ```bash
   # Create an alias for quick access
   alias wikitalk='cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation && source py314_venv/bin/activate && python wikitalk.py'
   
   # Then just type: wikitalk
   ```

3. **Customize the model:**
   Edit `config.py`:
   ```python
   # Current (384 dims, fast)
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   
   # Larger (768 dims, slower but more accurate)
   # EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
   ```
   Then rebuild index.

## FAQ

**Q: Can I rebuild the index without waiting 2 hours?**  
A: No, embedding generation requires processing 33.5M chunks. But you only need to do it once.

**Q: Does embedding search use GPU?**  
A: Yes, automatically if available. Check with `python -c "import torch; print(torch.cuda.is_available())"`

**Q: Can I use embedding search without the LLM?**  
A: Yes! You'll get search results but no AI-generated responses. Both work independently.

**Q: What if I need to rebuild the index?**  
A: Just run `python build_embeddings.py` again. It will overwrite the existing index.

**Q: How much disk space does the index take?**  
A: ~15GB for the FAISS index + ~200MB for ID mapping = ~15.2GB total.

## Success Checklist

✅ Database loaded (81.0 GB)  
✅ Embeddings built (33.5M vectors, ~15GB)  
✅ FAISS index saved  
✅ WikiTalk starts without errors  
✅ Search returns results in < 1 second  
✅ LLM generates responses (if server running)  

## Performance Summary

| Operation | Time |
|-----------|------|
| Build index | 1-2 hours (one-time) |
| Query embedding | 0.02-0.05s |
| FAISS search | 0.05-0.10s |
| Reranking | 0.01-0.05s |
| **Total per query** | **0.1-0.2s** |

---

**Ready?** → `time python build_embeddings.py`

For detailed info, see [[EMBEDDING_SEARCH.md]]
