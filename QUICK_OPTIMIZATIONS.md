# Quick Optimizations for WikiTalk Performance

Copy-paste these patches to immediately improve response time. Each can be applied independently.

## 🔥 OPTIMIZATION #1: Disable Query Rewriting (Save 2-5 seconds!)

**Why**: Query rewriting calls the LLM unnecessarily and doubles LLM latency.

**Edit**: `/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/llm_client.py`

**Find this** (around line 30-40):
```python
def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Rewrite query based on conversation history"""
    logger.debug(f"📝 Query rewrite requested for: '{query}'")
    
    if not conversation_history:
        logger.debug("   No conversation history, returning original query")
        return query
```

**Replace with**:
```python
def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Rewrite query based on conversation history"""
    # OPTIMIZATION: Skip query rewriting to save 2-5 seconds
    # Quality impact: Minimal - queries are usually specific enough already
    return query
```

**Impact**: 2-5 seconds faster ⚡

---

## 🔥 OPTIMIZATION #2: Reduce Retrieved Sources (Save 1-2 seconds)

**Why**: Processing and generating responses for fewer sources is faster.

**Edit**: `/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/wikitalk.py`

**Find this** (around line 74-76):
```python
# Retrieve relevant sources using appropriate search method
print("Searching Wikipedia...")
search_method = "embedding" if (self.use_embeddings and self.retriever.faiss_index) else "like"
sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
```

**Replace with**:
```python
# Retrieve relevant sources using appropriate search method
print("Searching Wikipedia...")
search_method = "embedding" if (self.use_embeddings and self.retriever.faiss_index) else "like"
sources = self.retriever.search(rewritten_query, top_k=3, method=search_method)
```

**Impact**: 1-2 seconds faster ⚡

---

## 🔥 OPTIMIZATION #3: Skip Result Reranking (Save 0.5-1 second)

**Why**: Fuzzy matching on many results is expensive; top FAISS results are usually good.

**Edit**: `/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/retriever.py`

**Find this** (around line 389-410):
```python
def search(self, query: str, top_k: int = 20, method: str = "embedding") -> List[Dict[str, Any]]:
    """Unified search method
    ...
    """
    start_time = time.time()
    
    if method == "embedding":
        results = self.embedding_search(query, RETRIEVAL_TOPK)
    else:
        results = self.bm25_search(query, RETRIEVAL_TOPK)
    
    # Rerank top results
    reranked_results = self.rerank_results(query, results, top_k)
    
    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(reranked_results)} results")
    
    return reranked_results
```

**Replace with**:
```python
def search(self, query: str, top_k: int = 20, method: str = "embedding") -> List[Dict[str, Any]]:
    """Unified search method
    ...
    """
    start_time = time.time()
    
    if method == "embedding":
        results = self.embedding_search(query, RETRIEVAL_TOPK)
    else:
        results = self.bm25_search(query, RETRIEVAL_TOPK)
    
    # OPTIMIZATION: Skip expensive reranking - FAISS results are already good
    # just return top_k directly
    final_results = results[:top_k]
    
    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(final_results)} results")
    
    return final_results
```

**Impact**: 0.5-1 second faster ⚡

---

## 🔥 OPTIMIZATION #4: Use Keyword Search Instead of Embeddings (Save 1-2 seconds)

**Why**: SQL LIKE search is much faster than generating embeddings.

**Trade-off**: Lower search quality, but often sufficient.

**Edit**: `/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/wikitalk.py`

**Find this** (around line 19-32):
```python
class WikiTalk:
    def __init__(self, use_embeddings=True):
        """Initialize WikiTalk
        
        Args:
            use_embeddings: If True, use semantic search; if False, use keyword search
        """
        # Initialize with embedding search by default (if available)
        self.retriever = HybridRetriever(use_bm25_only=not use_embeddings)
```

**Change to**:
```python
class WikiTalk:
    def __init__(self, use_embeddings=False):  # <-- Changed from True to False
        """Initialize WikiTalk
        
        Args:
            use_embeddings: If True, use semantic search; if False, use keyword search
        """
        # Initialize with keyword search for faster responses
        self.retriever = HybridRetriever(use_bm25_only=not use_embeddings)
```

**Impact**: 1-2 seconds faster ⚡

---

## 🔥 OPTIMIZATION #5: Reduce Conversation Context (Save 0.5-1 second)

**Why**: Shorter conversation history = shorter prompts = faster LLM responses.

**Edit**: `/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/config.py`

**Find this** (around line 62):
```python
# Retrieval configuration
RETRIEVAL_TOPK = 40
MEMORY_TURNS = 8
```

**Replace with**:
```python
# Retrieval configuration
RETRIEVAL_TOPK = 40
MEMORY_TURNS = 2  # Reduced from 8 for faster response generation
```

**Impact**: 0.5-1 second faster ⚡

---

## Combined Quick Win: 4-8 seconds faster! 🚀

Apply **optimizations 1-3** together for the best balance:

1. Disable query rewriting (saves 2-5s)
2. Reduce sources to 3 (saves 1-2s) 
3. Skip reranking (saves 0.5s)

**Total expected improvement**: 3-8 seconds faster

**Command to test**:
```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate
time python profile_wikitalk.py
```

---

## Undo Changes If Needed

All these changes are in non-critical paths. To revert:

1. **Disable query rewriting**: Add back the LLM call in `query_rewrite()`
2. **Sources to 3**: Change back to `top_k=5`
3. **Skip reranking**: Add back `rerank_results()` call
4. **Keyword search**: Change back `use_embeddings=True`

---

## What NOT to Optimize

❌ Don't modify FAISS/embedding generation itself - it's already optimized
❌ Don't change database schema - it's already indexed
❌ Don't reduce top_k below 3 - search quality degrades too much
❌ Don't disable the database - it's needed for metadata

These are all already fast (< 1 second each).

---

## Testing

After applying patches, test with:

```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate
python wikitalk.py
```

Ask a question and watch the `Processed in X.XXs` time.

**Target**: < 3 seconds per query

**Ideal**: < 2 seconds per query
