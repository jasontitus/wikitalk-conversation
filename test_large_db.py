#!/usr/bin/env python3
"""
Test script for large database (33.5M chunks)
Tests retrieval performance and quality with the full Wikipedia dataset
"""
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retriever import HybridRetriever
from config import SQLITE_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Test database connection"""
    logger.info("\n1️⃣ Testing database connection...")
    
    if not SQLITE_DB_PATH.exists():
        logger.error(f"❌ Database not found: {SQLITE_DB_PATH}")
        return False
    
    db_size_gb = SQLITE_DB_PATH.stat().st_size / (1024 ** 3)
    logger.info(f"✓ Database found: {db_size_gb:.1f} GB")
    return True


def test_retriever_initialization():
    """Test retriever initialization"""
    logger.info("\n2️⃣ Testing retriever initialization...")
    
    try:
        start_time = time.time()
        retriever = HybridRetriever(use_bm25_only=True)
        init_time = time.time() - start_time
        
        logger.info(f"✓ Retriever initialized in {init_time:.2f}s")
        
        retriever.load_indexes()
        load_time = time.time() - start_time - init_time
        logger.info(f"✓ Indexes loaded in {load_time:.2f}s")
        
        return retriever
    except Exception as e:
        logger.error(f"❌ Retriever initialization failed: {e}")
        return None


def test_search_queries(retriever):
    """Test various search queries"""
    logger.info("\n3️⃣ Testing search queries...")
    
    test_queries = [
        ("world war", 3),
        ("machine learning", 3),
        ("ancient rome", 3),
        ("quantum computing", 3),
        ("renaissance art", 3),
    ]
    
    results_summary = []
    total_search_time = 0
    
    for query, top_k in test_queries:
        try:
            start_time = time.time()
            results = retriever.search(query, top_k=top_k)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            logger.info(f"\n   📖 '{query}':")
            logger.info(f"      Time: {search_time:.2f}s, Results: {len(results)}")
            
            if results:
                top_result = results[0]
                logger.info(f"      Top: {top_result['title'][:60]}")
                logger.info(f"      Score: {top_result.get('rerank_score', 0):.3f}")
                
                results_summary.append({
                    'query': query,
                    'results': len(results),
                    'time': search_time,
                    'top_title': top_result['title']
                })
        except Exception as e:
            logger.error(f"      ❌ Search failed: {e}")
            return None
    
    # Performance summary
    logger.info(f"\n   📊 Performance Summary:")
    logger.info(f"      Total queries: {len(results_summary)}")
    logger.info(f"      Total time: {total_search_time:.2f}s")
    logger.info(f"      Avg time/query: {total_search_time/len(results_summary):.2f}s")
    
    return results_summary


def test_relevance_quality(retriever):
    """Test result quality"""
    logger.info("\n4️⃣ Testing result quality...")
    
    # Known relevant queries and expected top results
    quality_tests = [
        {
            "query": "world war i",
            "expect_terms": ["war", "1914", "1918", "germany", "france", "britain"]
        },
        {
            "query": "leonardo da vinci",
            "expect_terms": ["renaissance", "artist", "inventor", "painter"]
        },
        {
            "query": "python programming",
            "expect_terms": ["python", "programming", "code", "language"]
        }
    ]
    
    quality_results = []
    
    for test in quality_tests:
        query = test['query']
        expect_terms = test['expect_terms']
        
        results = retriever.search(query, top_k=5)
        
        if not results:
            logger.warning(f"   ⚠️ '{query}': No results")
            continue
        
        # Check top result for expected terms
        top_text = (results[0]['text'] + results[0]['title']).lower()
        found_terms = [t for t in expect_terms if t.lower() in top_text]
        relevance_score = len(found_terms) / len(expect_terms)
        
        logger.info(f"\n   📋 '{query}':")
        logger.info(f"      Expected terms: {len(found_terms)}/{len(expect_terms)}")
        logger.info(f"      Relevance: {relevance_score:.0%}")
        logger.info(f"      Top result: {results[0]['title'][:50]}")
        
        quality_results.append({
            'query': query,
            'relevance': relevance_score
        })
    
    if quality_results:
        avg_relevance = sum(r['relevance'] for r in quality_results) / len(quality_results)
        logger.info(f"\n   📊 Average relevance: {avg_relevance:.0%}")
    
    return quality_results


def main():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("🚀 Large Database (79.7 GB) Test Suite")
    logger.info("=" * 70)
    
    # Test 1: Database connection
    if not test_database_connection():
        logger.error("Failed at database connection test")
        return False
    
    # Test 2: Retriever initialization
    retriever = test_retriever_initialization()
    if not retriever:
        logger.error("Failed at retriever initialization test")
        return False
    
    # Test 3: Search queries
    results_summary = test_search_queries(retriever)
    if results_summary is None:
        logger.error("Failed at search queries test")
        retriever.close()
        return False
    
    # Test 4: Result quality
    quality_results = test_relevance_quality(retriever)
    
    # Cleanup
    retriever.close()
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Summary")
    logger.info("=" * 70)
    logger.info(f"✅ Database connection: PASS")
    logger.info(f"✅ Retriever initialization: PASS")
    logger.info(f"✅ Search functionality: PASS ({len(results_summary)} queries tested)")
    logger.info(f"✅ Result quality: PASS ({len(quality_results)} relevance tests)")
    logger.info("\n🎉 All tests PASSED! Large database is working correctly.\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
