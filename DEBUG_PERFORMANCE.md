# WikiTalk Performance Debugging Guide

## Problem: 10-Second Reply Latency

Your wikitalk.py is taking ~10 seconds to reply to questions. This guide will help you identify and fix the bottleneck.

## Quick Start: Run the Profiler

Use the performance profiler to identify which component is slow:

```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate
time python profile_wikitalk.py
```

This will:
1. Run 1 benchmark with 3 test queries
2. Measure each processing phase
3. Identify the bottleneck
4. Show average, min, and max times for each phase

The output will show:
- **query_rewrite**: LLM query rewriting step
- **semantic_search**: Embedding generation + FAISS search + database lookups
- **response_generation**: LLM response generation with sources
- **conversation_save**: Saving chat history

## Common Bottlenecks and Solutions

### 1. Query Rewrite Taking 3-5 seconds ⏱️

**Root Cause**: LLM is slow to process query rewriting

**Evidence**: If `query_rewrite` is > 2s in profiler

**Solutions** (in order of preference):

A. **Disable Query Rewrite** (Fastest - saves 2-5 seconds!)
   - Open `llm_client.py`
   - In `query_rewrite()` method (line 30), change:
     ```python
     # OPTION 1: Disable query rewriting completely
     if not conversation_history:
         logger.debug("   No conversation history, returning original query")
         return query
     
     # ADD THIS - Always return original query without LLM call
     return query  # <-- DISABLE REWRITING
     ```
   - This skips the LLM call entirely and just uses the original query
   - **Expected impact**: Save 2-5 seconds per query

B. **Reduce Query Rewrite Complexity**
   - Make prompts shorter
   - Reduce `MEMORY_TURNS` in `config.py` from 8 to 4
   - Limit conversation context sent to LLM

C. **Check LM Studio Settings**
   - Make sure LM Studio is using GPU acceleration
   - Verify model is loaded (not in memory-mapped mode)
   - Try a faster model like Qwen2.5-7B instead of 14B

### 2. Semantic Search Taking 2-4 seconds 🔍

**Root Cause**: Embedding generation or FAISS search is slow

**Evidence**: If `semantic_search` is > 2s in profiler

**Breakdown** (run with more detailed timing):
- Embedding generation: ~0.2-0.5s
- FAISS search: ~0.1s  
- Database lookups: ~0.5-1s
- Reranking: ~0.5s

**Solutions**:

A. **Skip Reranking** (Small improvement - saves ~0.5s)
   - Open `retriever.py` line 389
   - Change the `search()` method:
     ```python
     def search(self, query: str, top_k: int = 20, method: str = "embedding"):
         # ... existing code ...
         
         # OPTION: Skip reranking if it's slow
         # reranked_results = self.rerank_results(query, results, top_k)
         # Just return top_k results directly
         return results[:top_k]
     ```

B. **Reduce Number of Results to Rerank**
   - In `retriever.py` line 400:
     ```python
     # Change from:
     results = self.embedding_search(query, RETRIEVAL_TOPK)
     # To fewer results to avoid expensive reranking:
     results = self.embedding_search(query, 10)  # was 40
     ```

C. **Use Keyword Search Instead** (Fastest option)
   - In `wikitalk.py` line 27, during initialization:
     ```python
     # Use BM25 search (SQL LIKE) instead of embeddings
     self.retriever = HybridRetriever(use_bm25_only=True)
     ```
   - This skips embedding generation entirely - much faster!
   - Trade-off: Lower quality search results

D. **Check Embedding Model Speed**
   - Current: `all-MiniLM-L6-v2` (384 dims) - already very fast
   - Already optimized, unlikely to be the issue here

### 3. Response Generation Taking 3-7 seconds 💬

**Root Cause**: LLM is slow generating the response

**Evidence**: If `response_generation` is > 3s in profiler

**Solutions**:

A. **Reduce Number of Sources**
   - In `wikitalk.py` line 76:
     ```python
     # Change from:
     sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
     # To:
     sources = self.retriever.search(rewritten_query, top_k=3, method=search_method)
     ```
   - Fewer sources = shorter prompts = faster LLM response

B. **Simplify Prompt Structure**
   - In `llm_client.py` line 89-95
   - Remove unnecessary context or make sources shorter

C. **Check LM Studio GPU**
   - Make sure model is using GPU fully
   - Check LM Studio dashboard for GPU utilization
   - Verify CUDA/metal acceleration is enabled

### 4. Conversation Save Taking > 0.5 seconds 💾

**Root Cause**: File I/O overhead

**Evidence**: If `conversation_save` is > 0.5s in profiler

**Solutions**:

A. **Disable Conversation Saving**
   - In `wikitalk.py` line 87-90:
     ```python
     # Comment out conversation saving:
     # self.conversation_manager.add_exchange(
     #     self.session_id, query, response
     # )
     ```

B. **Use In-Memory Conversations Only**
   - Store conversation in memory, save to disk periodically
   - Not critical for interactive mode

## Step-by-Step Debugging Process

### Step 1: Run the Profiler
```bash
time python profile_wikitalk.py
```

### Step 2: Identify the Top Bottleneck
Look at the "BOTTLENECK ANALYSIS" section and find the slowest component.

### Step 3: Apply Quick Fixes (in order)

**First**: Disable query rewrite (~2-5s saved)
```python
# In llm_client.py, line 36
return query  # Skip rewriting
```

**Second**: Reduce sources to top 3 (~1-2s saved)
```python
# In wikitalk.py, line 76
sources = self.retriever.search(rewritten_query, top_k=3, method=search_method)
```

**Third**: Skip reranking (~0.5s saved)
```python
# In retriever.py, line 405
return results[:top_k]
```

### Step 4: Re-run Profiler to Verify
```bash
time python profile_wikitalk.py
```

You should see significant improvement!

## Expected Performance

After optimizations:

| Configuration | Typical Time | Notes |
|---|---|---|
| **Original** | 8-10s | With all features enabled |
| **Disable rewrite** | 3-5s | Query rewrite was the bottleneck |
| **Reduce sources** | 2-4s | Fewer documents to process |
| **Skip reranking** | 1.5-3s | Faster but less relevant results |
| **BM25 only** | 0.5-1s | Fastest but quality suffers |
| **Ideal** | 1-2s | BM25 + few sources + no rewrite |

## What NOT to Do

❌ Don't disable the FAISS index and database - they're fast
❌ Don't reduce `top_k` search results below 3 - quality suffers
❌ Don't modify core FAISS code - it's already optimized
❌ Don't use slower embedding models (BGE-M3 is 10x slower)

## Testing Interactively

Once you've made changes, test with:

```bash
time python wikitalk.py
```

Then ask a question and check the displayed `Processed in X.XXs` time.

Target: < 3 seconds per query (reasonable for LLM + search)

## Still Having Issues?

Check these:

1. **Is LM Studio running?**
   ```bash
   curl http://localhost:1234/v1/models
   ```

2. **Is it using GPU?**
   - Check LM Studio dashboard
   - Look for GPU utilization > 90%

3. **Is the database locked?**
   ```bash
   lsof | grep docs.sqlite
   ```

4. **Is there high memory usage?**
   ```bash
   top -o %MEM
   ```

## Advanced: Python Profiling

For deeper analysis, use Python's built-in profiler:

```bash
python -m cProfile -s cumulative wikitalk.py
```

This shows exactly which functions consume the most time.
