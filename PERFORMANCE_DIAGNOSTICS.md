# WikiTalk Performance Diagnostics - Quick Reference

## Your Problem
WikiTalk is taking ~10 seconds to reply to questions.

## Root Causes (Most Likely)

This is almost certainly **one of these**:

### 1. **Query Rewriting (2-5 seconds)** ← Most Common!
- Calls LLM twice: once to rewrite query, once to generate response
- LLM latency is usually 1-2s per call
- Two calls = 2-4s of your 10-second wait

### 2. **Response Generation (3-7 seconds)**
- Depends entirely on LLM speed
- Processing sources and generating answer takes time

### 3. **Semantic Search (1-3 seconds)**
- Embedding generation: 0.2-0.5s
- FAISS search: 0.1s (fast!)
- Database lookups: 0.5-1s
- Result reranking: 0.5s

### 4. **Conversation Saving (0.1-0.5 seconds)**
- Usually not significant
- JSON file write overhead

## Quick Diagnosis (5 minutes)

### Step 1: Run the Profiler
```bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate
time python profile_wikitalk.py
```

### Step 2: Look for This Pattern
The output will show something like:

```
query_rewrite:           3.2s ← If this is high, that's your problem!
semantic_search:         1.5s
response_generation:     4.1s ← Or this
conversation_save:       0.2s
```

## The Fix (Based on Your Profiler Output)

### If query_rewrite > 2s
🔥 **Fastest Fix**: Disable query rewriting in `llm_client.py` line 30

Replace:
```python
def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Rewrite query based on conversation history"""
    logger.debug(f"📝 Query rewrite requested for: '{query}'")
    
    if not conversation_history:
        logger.debug("   No conversation history, returning original query")
        return query
    # ... rest of function ...
```

With:
```python
def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Rewrite query based on conversation history - DISABLED FOR SPEED"""
    return query
```

**Time saved**: 2-5 seconds ⚡

---

### If response_generation > 3s
**Check LM Studio**:
1. Is it running? `curl http://localhost:1234/v1/models`
2. Is GPU enabled? Check LM Studio dashboard
3. Try a faster model (Qwen2.5-7B instead of 14B)

**Alternative**: Reduce sources in `wikitalk.py` line 76:
```python
sources = self.retriever.search(rewritten_query, top_k=3, method=search_method)  # was 5
```

**Time saved**: 1-2 seconds ⚡

---

### If semantic_search > 2s
This means reranking or DB lookups are slow.

**Quick fix** in `retriever.py` line 405:
Replace:
```python
reranked_results = self.rerank_results(query, results, top_k)
return reranked_results
```

With:
```python
return results[:top_k]  # Skip expensive reranking
```

**Time saved**: 0.5-1 second ⚡

## Expected Results

| Optimization | Time Saved | Total |
|---|---|---|
| Original | - | 10s |
| Disable query rewriting | 2-5s | 5-8s |
| + Reduce sources | 1-2s | 3-6s |
| + Skip reranking | 0.5-1s | 2-5s |

**Target**: < 3 seconds total

## Verify Your Fix

```bash
python wikitalk.py
```

Ask a question and check: `Processed in X.XXs`

Should now be 2-4 seconds instead of 10 seconds! 🚀

## Commands You'll Need

Activate environment:
```
source py314_venv/bin/activate
```

Run profiler:
```
time python profile_wikitalk.py
```

Run interactive mode:
```
python wikitalk.py
```

See detailed docs: `DEBUG_PERFORMANCE.md` and `QUICK_OPTIMIZATIONS.md`

## Advanced: Get Detailed Timings

Add this to `wikitalk.py` process_query() method to see each step:

```python
import time
start = time.time()

# Phase 1
rewritten_query = self.llm_client.query_rewrite(query, history)
print(f"  Query rewrite: {time.time()-start:.2f}s")

# Phase 2
search_start = time.time()
sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
print(f"  Search: {time.time()-search_start:.2f}s")

# Phase 3
gen_start = time.time()
response = self.llm_client.generate_response(rewritten_query, sources, history)
print(f"  Generation: {time.time()-gen_start:.2f}s")

print(f"Total: {time.time()-start:.2f}s")
```

This shows exact timing for each component.

## Still Slow?

1. **Check if LM Studio is responsive**:
   ```bash
   curl http://localhost:1234/v1/models
   ```

2. **Profile individual functions**:
   ```bash
   python -m cProfile -s cumulative wikitalk.py
   ```

3. **Check system resources**:
   ```bash
   top -o %CPU
   ```

## Don't Forget

✅ Disable query rewriting first (biggest impact)
✅ Apply one change at a time and re-test
✅ Check profiler output to confirm which component was slow
✅ Verify LM Studio is running with GPU enabled

You should see 4-8 seconds improvement total. Happy optimizing! 🎉
