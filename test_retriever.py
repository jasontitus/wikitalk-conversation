#!/usr/bin/env python3
"""
Test retriever with SQLite BM25 search
"""
import sqlite3
from pathlib import Path
from config import DATA_DIR

# Use test database instead of full database for quick testing
TEST_DB_PATH = DATA_DIR / "test_docs.sqlite"

def test_bm25_search(query, top_k=5):
    """Test BM25 search on SQLite database"""
    
    if not TEST_DB_PATH.exists():
        print(f"❌ SQLite test database not found at {TEST_DB_PATH}")
        print(f"Run: python test_data_processor.py")
        return
    
    print(f"\n🔍 Searching for: '{query}'")
    print(f"{'='*80}")
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Query the FTS5 index
        cursor.execute("""
            SELECT 
                c.id, c.text, c.title, c.url,
                rank
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
        
        for i, (chunk_id, text, title, url, rank) in enumerate(results, 1):
            score = 1.0 / (rank + 1)
            print(f"{i}. [{title}]")
            print(f"   Score: {score:.3f}")
            print(f"   Text: {text[:150]}...")
            if url:
                print(f"   URL: {url}")
            print()
    
    except Exception as e:
        print(f"❌ Error during search: {e}")
    
    finally:
        conn.close()

def test_database_info():
    """Show database statistics"""
    if not TEST_DB_PATH.exists():
        print(f"❌ SQLite test database not found")
        print(f"Run: python test_data_processor.py")
        return
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT title) FROM chunks")
        total_titles = cursor.fetchone()[0]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {total_chunks:,}")
        print(f"   Unique articles: {total_titles:,}")
    
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 WikiTalk Retriever Test")
    print("="*80)
    
    # Show database info
    test_database_info()
    
    # Test searches
    test_queries = [
        "World War I",
        "machine learning",
        "ancient Rome",
        "climate change",
        "Shakespeare"
    ]
    
    for query in test_queries:
        test_bm25_search(query, top_k=3)
