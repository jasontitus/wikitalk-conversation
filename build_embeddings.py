#!/usr/bin/env python3
"""
Build embedding index for WikiTalk
Creates FAISS index for semantic search on all 33.5M chunks
This takes 1-2 hours on a single GPU or 4-6 hours on CPU
"""

import logging
import sys
from pathlib import Path
import time
from datetime import timedelta
import psutil

sys.path.insert(0, str(Path(__file__).parent))

from retriever import HybridRetriever
from config import SQLITE_DB_PATH, FAISS_INDEX_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build embedding index"""
    logger.info("=" * 70)
    logger.info("🚀 Building Embedding Index for WikiTalk")
    logger.info("=" * 70)
    
    if not SQLITE_DB_PATH.exists():
        logger.error(f"❌ Database not found: {SQLITE_DB_PATH}")
        return False
    
    db_size_gb = SQLITE_DB_PATH.stat().st_size / (1024 ** 3)
    logger.info(f"\n📊 Database: {db_size_gb:.1f} GB")
    
    # System info
    process = psutil.Process()
    mem_total = psutil.virtual_memory().total / (1024 ** 3)
    mem_available = psutil.virtual_memory().available / (1024 ** 3)
    logger.info(f"💾 System RAM: {mem_total:.1f} GB total, {mem_available:.1f} GB available")
    
    # Check if index already exists
    if FAISS_INDEX_PATH.exists():
        logger.info(f"✓ FAISS index already exists: {FAISS_INDEX_PATH}")
        logger.info(f"   Size: {FAISS_INDEX_PATH.stat().st_size / (1024 ** 3):.1f} GB")
        
        response = input("\n❓ Rebuild index? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Skipping rebuild")
            return True
    
    logger.info("\n🔧 Initializing retriever with embedding building...")
    logger.info("   This will:")
    logger.info("   1. Stream all 33.5M chunks from database")
    logger.info("   2. Generate embeddings in batches")
    logger.info("   3. Create FAISS index")
    logger.info("   4. Save index to disk")
    
    logger.info("\n⏱️  Estimated time:")
    logger.info("   - GPU (NVIDIA): 1-2 hours")
    logger.info("   - GPU (M3 Max): 4-6 hours")
    logger.info("   - CPU: 6-12 hours")
    
    logger.info("\n📊 Memory prediction:")
    logger.info("   - Model load: ~2-3 GB")
    logger.info("   - Index building: 10-20 MB per batch")
    logger.info("   - Peak usage: ~30-50 GB (should be fine with 128GB)")
    logger.info("   - Final index: ~15 GB")
    
    response = input("\n❓ Continue? (y/n): ").strip().lower()
    if response != 'y':
        logger.info("Cancelled")
        return False
    
    # Monitor during build
    logger.info("\n" + "=" * 70)
    logger.info("📈 Build Progress Monitor")
    logger.info("=" * 70)
    logger.info("   Progress updates every 30 seconds or every 100K chunks")
    logger.info("   Watch memory and ETA to estimate completion time")
    logger.info("=" * 70 + "\n")
    
    # Build index
    start_time = time.time()
    start_mem = process.memory_info().rss / (1024 ** 3)
    
    try:
        retriever = HybridRetriever(use_bm25_only=False, build_embeddings=True)
        retriever.load_indexes()
        retriever.close()
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        end_mem = process.memory_info().rss / (1024 ** 3)
        peak_mem = end_mem  # Approximate peak
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Embedding index built successfully!")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total time: {elapsed_str}")
        logger.info(f"💾 Memory usage: {start_mem:.1f}GB → {end_mem:.1f}GB")
        logger.info(f"📁 Index saved to: {FAISS_INDEX_PATH}")
        logger.info(f"📊 Index size: {FAISS_INDEX_PATH.stat().st_size / (1024 ** 3):.1f} GB")
        
        logger.info("\n🎯 You can now use semantic search:")
        logger.info("   python wikitalk.py")
        logger.info("   User: Tell me about ancient roman architecture")
        logger.info("   WikiTalk: (will search semantically for relevant articles)")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Build interrupted by user")
        logger.warning("   You can resume by running the script again")
        return False
    except Exception as e:
        logger.error(f"\n❌ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
