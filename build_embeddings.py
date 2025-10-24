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
    logger.info("   - CPU: 4-6 hours")
    logger.info("   - Mac CPU: 6-12 hours")
    
    response = input("\n❓ Continue? (y/n): ").strip().lower()
    if response != 'y':
        logger.info("Cancelled")
        return False
    
    # Build index
    start_time = time.time()
    
    try:
        retriever = HybridRetriever(use_bm25_only=False, build_embeddings=True)
        retriever.load_indexes()
        retriever.close()
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Embedding index built successfully!")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total time: {elapsed_str}")
        logger.info(f"📁 Index saved to: {FAISS_INDEX_PATH}")
        logger.info(f"📊 Index size: {FAISS_INDEX_PATH.stat().st_size / (1024 ** 3):.1f} GB")
        
        logger.info("\n🎯 You can now use semantic search:")
        logger.info("   python wikitalk.py")
        logger.info("   User: Tell me about ancient roman architecture")
        logger.info("   WikiTalk: (will search semantically for relevant articles)")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Build interrupted by user")
        return False
    except Exception as e:
        logger.error(f"\n❌ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
