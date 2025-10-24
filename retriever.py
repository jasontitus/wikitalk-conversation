"""
Hybrid retrieval system for WikiTalk - Optimized for Large Database
"""
import sqlite3
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import json
import time

from config import *

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, use_bm25_only=True):
        """Initialize retriever - optimized for large databases
        
        Args:
            use_bm25_only: If True, use only BM25 search (recommended for large DB)
        """
        self.use_bm25_only = use_bm25_only
        self.embedding_model = None if use_bm25_only else SentenceTransformer(EMBEDDING_MODEL)
        self.faiss_index = None
        self.id_mapping = {}
        self.conn = None
        
        # Connection pool for better performance
        self._init_db_connection()
        
    def _init_db_connection(self):
        """Initialize database connection with optimizations"""
        try:
            self.conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=30)
            self.conn.row_factory = sqlite3.Row  # Return dict-like rows
            
            # Enable query optimizations
            self.conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging for better concurrency
            self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster I/O
            self.conn.execute("PRAGMA cache_size=10000")  # Larger cache
            self.conn.execute("PRAGMA query_only=true")  # Read-only mode for safety
            
            logger.info("Database connection initialized with optimizations")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        
    def load_indexes(self):
        """Load retrieval indexes"""
        logger.info("Loading retrieval indexes...")
        
        if self.use_bm25_only:
            logger.info("Using BM25 search only (optimized for large database)")
            # Verify SQLite has FTS5 index
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
            if not cursor.fetchone():
                logger.warning("FTS5 index not found. Full-text search may be slow.")
            logger.info("Indexes loaded successfully")
            return
        
        # Load FAISS index if hybrid search is enabled
        try:
            if FAISS_INDEX_PATH.exists():
                logger.info("Loading FAISS index...")
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(IDS_MAPPING_PATH, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                logger.info(f"FAISS index loaded with {len(self.id_mapping)} vectors")
            else:
                logger.warning("FAISS index not found. Using BM25 only.")
                self.use_bm25_only = True
        except Exception as e:
            logger.warning(f"Failed to load FAISS: {e}. Using BM25 only.")
            self.use_bm25_only = True
        
        logger.info("Indexes loaded successfully")
    
    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """BM25 search using SQLite FTS5 - Optimized for large databases"""
        start_time = time.time()
        
        try:
            cursor = self.conn.cursor()
            
            # Improved FTS5 query with better ranking
            cursor.execute("""
                SELECT 
                    c.id, c.text, c.title, c.page_id, c.url, c.date_modified,
                    c.wikidata_id, c.infoboxes, c.has_math, c.start_pos, c.end_pos,
                    rank
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'text': row[1],
                    'title': row[2],
                    'page_id': row[3],
                    'url': row[4],
                    'date_modified': row[5],
                    'wikidata_id': row[6],
                    'infoboxes': row[7],
                    'has_math': row[8],
                    'start_pos': row[9],
                    'end_pos': row[10],
                    'rank': row[11],
                    'score': 1.0 / (row[11] + 1)  # Convert rank to score
                })
            
            elapsed = time.time() - start_time
            logger.debug(f"BM25 search completed in {elapsed:.2f}s, found {len(results)} results")
            
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def dense_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """Dense vector search using FAISS - Only if available"""
        if self.faiss_index is None or self.use_bm25_only:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            cursor = self.conn.cursor()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                    
                chunk_id = self.id_mapping[idx]
                
                # Get full metadata from SQLite
                cursor.execute("""
                    SELECT id, text, title, page_id, url, date_modified,
                           wikidata_id, infoboxes, has_math, start_pos, end_pos
                    FROM chunks WHERE id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'title': row[2],
                        'page_id': row[3],
                        'url': row[4],
                        'date_modified': row[5],
                        'wikidata_id': row[6],
                        'infoboxes': row[7],
                        'has_math': row[8],
                        'start_pos': row[9],
                        'end_pos': row[10],
                        'score': float(score)
                    })
            
            return results
        except Exception as e:
            logger.warning(f"Dense search failed, falling back to BM25: {e}")
            return []
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """Rerank results using RapidFuzz"""
        for result in results:
            # Calculate fuzzy match score
            text_score = fuzz.partial_ratio(query.lower(), result['text'].lower())
            title_score = fuzz.partial_ratio(query.lower(), result['title'].lower())
            
            # Combine with original score
            result['rerank_score'] = (
                result.get('score', 0) * 0.7 + 
                text_score * 0.2 + 
                title_score * 0.1
            ) / 100
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Unified search method - Uses best available method
        
        For large databases (33.5M chunks), BM25 is recommended
        """
        start_time = time.time()
        
        if self.use_bm25_only or self.faiss_index is None:
            # BM25 only search
            logger.debug(f"Using BM25 search for: {query}")
            results = self.bm25_search(query, RETRIEVAL_TOPK)
        else:
            # Hybrid search
            logger.debug(f"Using hybrid search for: {query}")
            
            # Get results from both methods
            bm25_results = self.bm25_search(query, RETRIEVAL_TOPK)
            dense_results = self.dense_search(query, RETRIEVAL_TOPK)
            
            # Combine and deduplicate
            all_results = {}
            
            # Add BM25 results
            for result in bm25_results:
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                    all_results[chunk_id]['bm25_score'] = result['score']
                    all_results[chunk_id]['dense_score'] = 0.0
                else:
                    all_results[chunk_id]['bm25_score'] = result['score']
            
            # Add dense results
            for result in dense_results:
                chunk_id = result['id']
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                    all_results[chunk_id]['bm25_score'] = 0.0
                    all_results[chunk_id]['dense_score'] = result['score']
                else:
                    all_results[chunk_id]['dense_score'] = result['score']
            
            # Combine scores
            combined_results = []
            for result in all_results.values():
                bm25_score = result.get('bm25_score', 0)
                dense_score = result.get('dense_score', 0)
                result['combined_score'] = 0.6 * dense_score + 0.4 * bm25_score
                combined_results.append(result)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            results = combined_results[:top_k * 2]
        
        # Rerank top results
        reranked_results = self.rerank_results(query, results, top_k)
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(reranked_results)} results")
        
        return reranked_results
    
    def format_sources(self, results: List[Dict[str, Any]]) -> str:
        """Format results as source citations"""
        sources = []
        for i, result in enumerate(results, 1):
            source = f"[{i}] {result['title']}"
            if result.get('url'):
                source += f" ({result['url']})"
            sources.append(source)
        return "\n".join(sources)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing HybridRetriever with large database")
    
    # Initialize with BM25-only mode for large database
    retriever = HybridRetriever(use_bm25_only=True)
    retriever.load_indexes()
    
    # Test search
    test_queries = [
        "World War I causes",
        "machine learning algorithms",
        "ancient Rome history"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=3)
        
        logger.info(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['title']} (score: {result.get('rerank_score', 0):.3f})")
            logger.info(f"   {result['text'][:150]}...")
    
    retriever.close()

