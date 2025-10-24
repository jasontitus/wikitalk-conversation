#!/usr/bin/env python3
"""
Simple test retriever - uses basic SQL LIKE queries without FTS5
"""
import sqlite3
from pathlib import Path
from config import DATA_DIR

TEST_DB_PATH = DATA_DIR / "test_docs.sqlite"

def simple_search(query, top_k=5):
    """Simple text search using LIKE"""
    if not TEST_DB_PATH.exists():
        print(f"❌ Test database not found: {TEST_DB_PATH}")
        print("Run: python test_simple_data.py")
        return
    
    print(f"\n🔍 Searching for: '{query}'")
    print(f"{'='*80}")
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Simple LIKE search
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT title, text
            FROM chunks
            WHERE title LIKE ? OR text LIKE ?
            LIMIT ?
        """, (search_term, search_term, top_k))
        
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

def show_stats():
    """Show database statistics"""
    if not TEST_DB_PATH.exists():
        print(f"❌ Test database not found")
        print("Run: python test_simple_data.py")
        return
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT title) FROM chunks")
        articles = cursor.fetchone()[0]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {total:,}")
        print(f"   Unique articles: {articles:,}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 WikiTalk Simple Test Retriever")
    print("="*80)
    
    show_stats()
    
    test_queries = [
        "World War",
        "machine learning",
        "ancient Rome",
        "climate",
        "Shakespeare"
    ]
    
    for query in test_queries:
        simple_search(query, top_k=3)
