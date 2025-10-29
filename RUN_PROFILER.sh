#!/bin/bash
cd /Users/jasontitus/experiments/wikiedia-conversation/wikipedia-conversation
source py314_venv/bin/activate
echo "🔧 Starting WikiTalk Performance Profiler..."
echo "This will help identify which component is causing the 10-second latency."
echo ""
time python profile_wikitalk.py
