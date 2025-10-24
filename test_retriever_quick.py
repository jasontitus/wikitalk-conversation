#!/usr/bin/env python3
"""
Quick test of FTS5 search on test database (6.7M chunks from 3 parquet files)
"""
import sqlite3
from pathlib import Path
from config import DATA_DIR

TEST_DB_PATH = DATA_DIR / "test_docs.sqlite"

def test_search(query, top_k=5):
    """Test FTS5 search"""
    if not TEST_DB_PATH.exists():
        print(f"❌ Test database not found: {TEST_DB_PATH}")
        print("Run: python test_data_processor.py")
        return
    
    print(f"\n🔍 Searching for: '{query}'")
    print(f"{'='*80}")
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT c.title, c.text
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k))
        
        results = cursor.fetchall()
        
        if not results:
            print("⚠️  No results found")
            return
        
        print(f"✅ Found {len(results)} results:\n")
        
        for i, (title, text) in enumerate(results, 1):
            print(f"{i}. [{title}]")
            print(f"   {text[:150]}...")
            print()
    
    except Exception as e:
        print(f"❌ Error during search: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 WikiTalk Test Retriever (Quick)")
    print("="*80)
    
    # Test searches
    test_queries = [
        "ancient Rome",
        "Napoleon",
        "Shakespeare",
        "machine learning",
        "history"
    ]
    
    for query in test_queries:
        test_search(query, top_k=3)
