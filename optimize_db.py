#!/usr/bin/env python3
"""
Database optimization script
Analyzes and optimizes indexes for faster FTS5 searches
"""
import sqlite3
import logging
import sys
from pathlib import Path
import time

from config import SQLITE_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_health():
    """Check database health and get statistics"""
    logger.info("📊 Database Health Check")
    logger.info("=" * 70)
    
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=60)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        logger.info(f"Total chunks: {chunk_count:,}")
        
        cursor.execute("SELECT COUNT(DISTINCT title) FROM chunks")
        article_count = cursor.fetchone()[0]
        logger.info(f"Unique articles: {article_count:,}")
        
        # Check indexes
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name LIKE '%chunks%'
        """)
        indexes = cursor.fetchall()
        logger.info(f"Indexes on chunks table: {len(indexes)}")
        for idx in indexes:
            logger.info(f"  - {idx[0]}")
        
        # Check FTS5 table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='chunks_fts'
        """)
        if cursor.fetchone():
            logger.info("✓ FTS5 index table found: chunks_fts")
        else:
            logger.warning("⚠️ FTS5 index table NOT found")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def optimize_database():
    """Optimize database for faster queries"""
    logger.info("\n🔧 Database Optimization")
    logger.info("=" * 70)
    
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=120)
        cursor = conn.cursor()
        
        # 1. Analyze table
        logger.info("\n1️⃣ Running ANALYZE...")
        start = time.time()
        cursor.execute("ANALYZE")
        conn.commit()
        logger.info(f"   ✓ ANALYZE completed in {time.time() - start:.2f}s")
        
        # 2. Check for missing indexes
        logger.info("\n2️⃣ Checking indexes...")
        
        # Add index on page_id if not exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='idx_chunks_page_id'
        """)
        if not cursor.fetchone():
            logger.info("   Adding index on page_id...")
            start = time.time()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)")
            conn.commit()
            logger.info(f"   ✓ Index created in {time.time() - start:.2f}s")
        else:
            logger.info("   ✓ page_id index exists")
        
        # Add index on title if not exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='idx_chunks_title'
        """)
        if not cursor.fetchone():
            logger.info("   Adding index on title...")
            start = time.time()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title)")
            conn.commit()
            logger.info(f"   ✓ Index created in {time.time() - start:.2f}s")
        else:
            logger.info("   ✓ title index exists")
        
        # 3. Optimize PRAGMA settings
        logger.info("\n3️⃣ Optimizing PRAGMA settings...")
        
        cursor.execute("PRAGMA optimize")
        conn.commit()
        logger.info("   ✓ PRAGMA optimize completed")
        
        # 4. Vacuum (optional, for space)
        logger.info("\n4️⃣ Vacuuming database (this may take a while)...")
        start = time.time()
        cursor.execute("VACUUM")
        elapsed = time.time() - start
        logger.info(f"   ✓ VACUUM completed in {elapsed:.2f}s")
        
        # Get final statistics
        db_size = SQLITE_DB_PATH.stat().st_size / (1024 ** 3)
        logger.info(f"\n📊 Final database size: {db_size:.2f} GB")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False


def test_fts5_performance():
    """Test FTS5 query performance"""
    logger.info("\n⚡ FTS5 Performance Test")
    logger.info("=" * 70)
    
    test_queries = [
        "world war",
        "machine learning",
        "ancient rome",
        "quantum physics",
        "renaissance"
    ]
    
    try:
        conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=60)
        cursor = conn.cursor()
        
        total_time = 0
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            start = time.time()
            cursor.execute("""
                SELECT c.id, c.title
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY f.rank
                LIMIT 10
            """, (query,))
            
            results = cursor.fetchall()
            elapsed = time.time() - start
            total_time += elapsed
            
            logger.info(f"  Time: {elapsed:.2f}s, Results: {len(results)}")
            if results:
                logger.info(f"  Top: {results[0][1][:60]}")
        
        avg_time = total_time / len(test_queries)
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   Total queries: {len(test_queries)}")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average time/query: {avg_time:.2f}s")
        
        if avg_time < 2.0:
            logger.info("   ✅ Performance is good!")
        elif avg_time < 5.0:
            logger.warning("   ⚠️ Performance is acceptable")
        else:
            logger.error("   ❌ Performance is slow - needs more optimization")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False


def main():
    """Run all optimization steps"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 WikiTalk Database Optimization Tool")
    logger.info("=" * 70)
    
    if not SQLITE_DB_PATH.exists():
        logger.error(f"Database not found: {SQLITE_DB_PATH}")
        return False
    
    # Check health
    if not check_database_health():
        return False
    
    # Optimize
    if not optimize_database():
        return False
    
    # Test performance
    if not test_fts5_performance():
        return False
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Database optimization complete!")
    logger.info("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
