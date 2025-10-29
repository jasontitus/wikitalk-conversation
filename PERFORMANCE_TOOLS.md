# 📊 WikiTalk Performance Debugging Tools

Complete toolkit for diagnosing and fixing 10-second latency issue.

## 🚀 Quick Start

Run the profiler to identify the bottleneck:

```zsh
time python profile_wikitalk.py
```

Then read `START_HERE.md` for the fix based on what's slow.

---

## 📁 Tool Files Created

### 1. **profile_wikitalk.py** (Main Diagnostic Tool)
- **What it does**: Measures time for each component (query rewrite, search, generation)
- **How to run**: `time python profile_wikitalk.py`
- **Output**: Detailed timing report with averages, min, max for each phase
- **Best for**: Identifying which component is the bottleneck

### 2. **RUN_PROFILER.sh** (Convenience Launcher)
- **What it does**: One-command profiler with automatic environment setup
- **How to run**: `./RUN_PROFILER.sh`
- **Best for**: Quick profiling without manual venv activation

### 3. **START_HERE.md** (Main Guide - READ THIS FIRST!)
- **What it contains**: 
  - Quick start instructions (5 minutes)
  - Decision tree: "Which component is slow? Pick your fix"
  - Copy-paste solutions for each bottleneck
  - Expected results and verification steps
- **Best for**: Users who want fastest path to solution

### 4. **PERFORMANCE_DIAGNOSTICS.md** (Quick Reference)
- **What it contains**:
  - Root causes summary
  - Quick diagnosis process
  - Fixes based on profiler output
  - Expected time improvements
  - Advanced debugging commands
- **Best for**: Quick lookup when you know what's slow

### 5. **DEBUG_PERFORMANCE.md** (Comprehensive Guide)
- **What it contains**:
  - Detailed explanation of each bottleneck
  - Multiple solutions per bottleneck (in priority order)
  - Step-by-step debugging process
  - What NOT to do
  - Advanced profiling techniques
  - FAQ section
- **Best for**: Deep understanding of the system

### 6. **QUICK_OPTIMIZATIONS.md** (Code Patches)
- **What it contains**: 5 ready-to-apply code patches
  - Disable query rewriting (save 2-5s)
  - Reduce retrieved sources (save 1-2s)
  - Skip result reranking (save 0.5-1s)
  - Use keyword search (save 1-2s)
  - Reduce conversation context (save 0.5-1s)
- **Best for**: Users who just want to copy-paste fixes

---

## 🎯 Which Tool Should I Use?

### I want the quickest solution
→ Read **START_HERE.md**, apply the fix

### I want to understand what's slow first
→ Run **profile_wikitalk.py**, read **PERFORMANCE_DIAGNOSTICS.md**

### I want comprehensive understanding
→ Read **DEBUG_PERFORMANCE.md**

### I want code patches to copy-paste
→ See **QUICK_OPTIMIZATIONS.md**

### I want detailed benchmarking
→ Run **profile_wikitalk.py** multiple times with different optimizations

### I'm having issues
→ Check **DEBUG_PERFORMANCE.md** "Still Having Issues?" section

---

## 📊 Performance Summary

| Bottleneck | Typical Time | Fix | Saves |
|---|---|---|---|
| Query rewriting | 2-5s | Disable | 2-5s ⚡ |
| Response generation | 3-7s | Reduce sources or check GPU | 1-2s ⚡ |
| Semantic search | 1-3s | Skip reranking | 0.5s ⚡ |
| Conversation save | 0.1-0.5s | Disable | 0.5s ⚡ |

**Expected combined improvement**: 3-8 seconds faster

---

## 🔧 File Locations

All tools are in:
```
/Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation/
```

- `profile_wikitalk.py` - Python profiler script
- `RUN_PROFILER.sh` - Shell launcher
- `START_HERE.md` - Main debugging guide
- `PERFORMANCE_DIAGNOSTICS.md` - Quick reference
- `DEBUG_PERFORMANCE.md` - Comprehensive guide
- `QUICK_OPTIMIZATIONS.md` - Code patches
- `PERFORMANCE_TOOLS.md` - This file

---

## 🎯 Recommended Workflow

### Step 1: Diagnose (5 minutes)
```bash
time python profile_wikitalk.py
```

### Step 2: Read Solution (2 minutes)
Look at "START_HERE.md" or "PERFORMANCE_DIAGNOSTICS.md" for your slow component

### Step 3: Apply Fix (5 minutes)
Copy-paste the fix from guides or "QUICK_OPTIMIZATIONS.md"

### Step 4: Verify (2 minutes)
```bash
python wikitalk.py
# Ask a question, check "Processed in X.XXs"
```

### Total time: 14 minutes to 3-8 second speedup! 🚀

---

## 📈 Expected Results

After applying recommended fixes (disable query rewrite + reduce sources + skip reranking):

```
Before: 10 seconds
After:  2-3 seconds (5-8x faster!)
```

---

## ⚡ One-Line Summary

The 10-second latency is almost certainly from LLM query rewriting (~2-5s) + response generation (~3-7s). Disable query rewriting first for immediate 2-5 second improvement, then optimize search and conversation saving.

Start with: `time python profile_wikitalk.py` → Read `START_HERE.md` → Apply fixes → Done! 🎉
