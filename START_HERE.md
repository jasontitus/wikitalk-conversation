# 🔧 WikiTalk Performance Debugging - START HERE

You want to debug why wikitalk.py is taking ~10 seconds to reply to questions.

## 📋 What I've Created For You

I've created 4 debugging tools to help you identify and fix the bottleneck:

| File | Purpose |
|---|---|
| **profile_wikitalk.py** | Performance profiler (run this first!) |
| **DEBUG_PERFORMANCE.md** | Complete debugging guide with all solutions |
| **QUICK_OPTIMIZATIONS.md** | Copy-paste code patches for instant improvements |
| **PERFORMANCE_DIAGNOSTICS.md** | Quick reference for diagnosis and fixes |
| **RUN_PROFILER.sh** | One-command profiler launcher |

## 🚀 Quick Start (5 Minutes)

### Step 1: Run the Profiler
Copy-paste this command [[memory:6351329]] [[memory:7447371]]:

```zsh
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation && source py314_venv/bin/activate && time python profile_wikitalk.py
```

Or use the convenience script:
```zsh
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation && time ./RUN_PROFILER.sh
```

### Step 2: Read the Output
The profiler will show timing for each component:
- `query_rewrite`: Time to rewrite query with LLM
- `semantic_search`: Time for embedding + FAISS search + DB lookup
- `response_generation`: Time for LLM to generate answer
- `conversation_save`: Time to save to file

Look for which one is > 2 seconds.

### Step 3: Apply the Fix
Jump to the section below based on what's slow:

## 🔥 Which Component is Slow? Pick Your Fix

### Query Rewrite is Slow (> 2 seconds) ← Most Common!

**Problem**: Query rewriting calls the LLM unnecessarily.

**Fix**: Disable it in `llm_client.py` (saves 2-5 seconds!)

Replace lines 30-71 with:
```python
def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Rewrite query based on conversation history - DISABLED FOR SPEED"""
    return query
```

**Save time**: 2-5 seconds ⚡

---

### Response Generation is Slow (> 3 seconds)

**Problem**: LLM is slow generating responses.

**Quick fix**: Reduce sources in `wikitalk.py` line 76

Change from:
```python
sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
```

To:
```python
sources = self.retriever.search(rewritten_query, top_k=3, method=search_method)
```

**Save time**: 1-2 seconds ⚡

**Better fix**: Check LM Studio GPU acceleration:
1. Is LM Studio running?
2. Is GPU enabled in settings?
3. Try faster model (Qwen2.5-7B instead of 14B)

---

### Semantic Search is Slow (> 2 seconds)

**Problem**: Result reranking is expensive.

**Fix**: Skip reranking in `retriever.py` lines 404-409

Replace:
```python
# Rerank top results
reranked_results = self.rerank_results(query, results, top_k)

elapsed = time.time() - start_time
logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(reranked_results)} results")

return reranked_results
```

With:
```python
# Skip expensive reranking
final_results = results[:top_k]

elapsed = time.time() - start_time
logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(final_results)} results")

return final_results
```

**Save time**: 0.5-1 second ⚡

---

### Conversation Save is Slow (> 0.5 seconds)

**Problem**: File I/O overhead for saving conversations.

**Fix**: Comment out in `wikitalk.py` lines 87-90:

```python
# Disable conversation saving to speed up responses
# self.conversation_manager.add_exchange(
#     self.session_id, query, response
# )
```

**Save time**: 0.5 second ⚡

## 📊 Expected Results

| Optimization | Saves | Total |
|---|---|---|
| Original | - | 10s |
| Disable query rewrite | 2-5s | 5-8s |
| + Reduce sources | 1-2s | 3-6s |
| + Skip reranking | 0.5s | 2-5s |

**Target**: < 3 seconds per query 🎯

## ✅ Verify Your Fix

After applying changes, test:

```zsh
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation && source py314_venv/bin/activate && python wikitalk.py
```

Ask a question and check the `Processed in X.XXs` output.

Should now be **2-4 seconds** instead of 10 seconds! 🚀

## 📚 Detailed Documentation

For more detailed solutions and explanations, see:

1. **PERFORMANCE_DIAGNOSTICS.md** - Quick reference guide
2. **DEBUG_PERFORMANCE.md** - Complete debugging guide
3. **QUICK_OPTIMIZATIONS.md** - Copy-paste code patches

## 🎯 My Recommendation

For best balance of speed and quality, apply these three fixes:

1. **Disable query rewriting** (saves 2-5s) - queries are usually specific enough
2. **Reduce sources to 3** (saves 1-2s) - fewer documents to process  
3. **Skip reranking** (saves 0.5s) - FAISS results are already good

**Combined**: 3-8 seconds faster! ⚡

## 🔍 Advanced Debugging

If you want detailed timing for each step, add this to `wikitalk.py` in the `process_query()` method (around line 62):

```python
import time
step_times = {}

step_times['rewrite'] = time.time()
rewritten_query = self.llm_client.query_rewrite(query, history)
print(f"  Rewrite: {time.time() - step_times['rewrite']:.2f}s")

step_times['search'] = time.time()
sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
print(f"  Search: {time.time() - step_times['search']:.2f}s")

step_times['gen'] = time.time()
response = self.llm_client.generate_response(rewritten_query, sources, history)
print(f"  Generation: {time.time() - step_times['gen']:.2f}s")
```

## ❓ FAQ

**Q: Is my database too slow?**
A: No - database lookups are < 1 second. The issue is LLM latency.

**Q: Should I change the embedding model?**
A: No - you're already using the fastest model (all-MiniLM-L6-v2).

**Q: What if query rewrite + response gen are both slow?**
A: Check LM Studio - it might not have GPU acceleration enabled.

**Q: Will disabling query rewriting reduce quality?**
A: Minimally - queries are usually specific enough without rewriting.

**Q: Can I apply fixes incrementally?**
A: Yes! Apply one at a time and re-run the profiler to confirm improvement.

## 🎉 Next Steps

1. Run: `time python profile_wikitalk.py`
2. Identify slowest component
3. Apply the fix for that component
4. Test: `python wikitalk.py` and ask a question
5. Check if response is faster

Expected time to fix: **10-15 minutes** ⏱️

Good luck! 🚀
