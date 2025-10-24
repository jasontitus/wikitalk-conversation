"""
Hybrid retrieval system for WikiTalk - Optimized for Large Database
Now with efficient embedding search using streaming
"""
import sqlite3
import pickle
import faiss
import numpy as np
import logging
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import json
import time

from config import *

logger = logging.getLogger(__name__)

# Detect and set device for GPU acceleration
def get_device():
    """Get the best available device for inference"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA available: True")
    else:
        device = "cpu"
        logger.warning("⚠ GPU not detected, using CPU (slower)")
    return device

DEVICE = get_device()


class HybridRetriever:
    def __init__(self, use_bm25_only=False, build_embeddings=False):
        """Initialize retriever
        
        Args:
            use_bm25_only: If True, use only LIKE search (fast fallback)
            build_embeddings: If True, build embedding index from scratch
        """
        self.use_bm25_only = use_bm25_only
        self.build_embeddings = build_embeddings
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE) if not use_bm25_only else None
        self.faiss_index = None
        self.id_mapping = {}
        self.conn = None
        self.article_embeddings = {}  # Cache of article embeddings
        
        # Connection pool for better performance
        self._init_db_connection()
        
    def _init_db_connection(self):
        """Initialize database connection with optimizations"""
        try:
            self.conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=30)
            self.conn.row_factory = sqlite3.Row
            
            # Enable query optimizations
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA cache_size=10000")
            self.conn.execute("PRAGMA query_only=true")
            
            logger.info("Database connection initialized with optimizations")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        
    def load_indexes(self):
        """Load retrieval indexes"""
        logger.info("Loading retrieval indexes...")
        
        if self.use_bm25_only:
            logger.info("Using LIKE search only (fast fallback)")
            return
        
        # Try to load existing FAISS index
        if FAISS_INDEX_PATH.exists() and not self.build_embeddings:
            try:
                logger.info("Loading existing FAISS index...")
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(IDS_MAPPING_PATH, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                logger.info(f"✓ FAISS index loaded with {len(self.id_mapping)} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Will rebuild.")
                self.build_embeddings = True
        else:
            self.build_embeddings = True
        
        # Build embeddings if needed
        if self.build_embeddings:
            self._build_embedding_index_streaming()
        
        logger.info("Indexes loaded successfully")
    
    def _build_embedding_index_streaming(self):
        """Build FAISS index by streaming chunks from database in batches"""
        logger.info("🚀 Building embedding index from database (streaming)...")
        logger.info("   This will take 1-2 hours for 33.5M chunks")
        
        # Optimized batch sizes for better GPU/CPU utilization
        embedding_batch_size = 1024  # Increased from 512 to maximize GPU
        db_batch_size = 10000  # Increased from 5000 to reduce I/O overhead
        
        try:
            import psutil
            import time as time_module
            
            cursor = self.conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            logger.info(f"   Total chunks to process: {total_chunks:,}")
            
            # Initialize FAISS index using IVFFlat (more efficient for large datasets)
            # IVFFlat partitions the space into buckets, avoiding slowdown as index grows
            logger.info("   Creating FAISS IVFFlat index (optimized for large datasets)...")
            nlist = 1000  # Number of partitions (buckets)
            quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist)
            self.id_mapping = {}
            vector_index = 0
            
            # Track timing and memory
            build_start_time = time_module.time()
            last_progress_time = build_start_time
            last_progress_chunks = 0
            process = psutil.Process()
            
            # IMPORTANT: IVFFlat needs training on sample data before adding vectors
            logger.info("   Training index on sample data...")
            train_samples = []
            cursor.execute("SELECT text, title FROM chunks LIMIT 40000")
            for row in cursor.fetchall():
                texts = [f"{row[1]}: {row[0][:500]}"]
                embeddings = self.embedding_model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                train_samples.append(embeddings[0])
            
            if train_samples:
                train_data = np.array(train_samples, dtype='float32')
                faiss.normalize_L2(train_data)
                self.faiss_index.train(train_data)
                logger.info(f"   ✓ Index trained on {len(train_samples)} samples")
            
            # Process in streaming batches
            processed = 0
            while processed < total_chunks:
                batch_start_time = time_module.time()
                
                # Read batch of chunks from database
                cursor.execute("""
                    SELECT id, text, title
                    FROM chunks
                    LIMIT ? OFFSET ?
                """, (db_batch_size, processed))
                
                chunk_batch = cursor.fetchall()
                if not chunk_batch:
                    break
                
                # Process embeddings in sub-batches (larger for better GPU utilization)
                for i in range(0, len(chunk_batch), embedding_batch_size):
                    sub_batch = chunk_batch[i:i+embedding_batch_size]
                    
                    # Create texts for embedding (combine title and text for context)
                    texts = [f"{row[2]}: {row[1][:500]}" for row in sub_batch]
                    
                    # Generate embeddings with larger batch for GPU efficiency
                    embeddings = self.embedding_model.encode(
                        texts,
                        batch_size=256,  # Increased from 64 to better utilize GPU
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    embeddings = embeddings.astype('float32')
                    faiss.normalize_L2(embeddings)
                    
                    # Add to FAISS index
                    self.faiss_index.add(embeddings)
                    
                    # Store ID mapping
                    for j, row in enumerate(sub_batch):
                        self.id_mapping[vector_index + j] = row[0]
                    
                    vector_index += len(sub_batch)
                
                processed += len(chunk_batch)
                current_time = time_module.time()
                
                # Log progress every 100K chunks or every 30 seconds
                if processed % 100000 == 0 or (current_time - last_progress_time) > 30:
                    elapsed = current_time - build_start_time
                    chunks_in_period = processed - last_progress_chunks
                    time_in_period = current_time - last_progress_time
                    
                    if time_in_period > 0:
                        chunks_per_sec = chunks_in_period / time_in_period
                        remaining_chunks = total_chunks - processed
                        eta_seconds = remaining_chunks / chunks_per_sec if chunks_per_sec > 0 else 0
                        eta_minutes = eta_seconds / 60
                        eta_hours = eta_minutes / 60
                    else:
                        chunks_per_sec = 0
                        eta_hours = 0
                    
                    # Get memory info
                    mem_info = process.memory_info()
                    mem_mb = mem_info.rss / (1024 * 1024)
                    percent_done = (processed / total_chunks) * 100
                    percent_ram = (mem_mb / (128 * 1024)) * 100  # Assuming 128GB system
                    
                    elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s"
                    
                    logger.info(f"")
                    logger.info(f"   ✓ Processed {processed:,}/{total_chunks:,} chunks ({percent_done:.1f}%)")
                    logger.info(f"     Time elapsed: {elapsed_str}")
                    logger.info(f"     Speed: {chunks_per_sec:.0f} chunks/sec")
                    logger.info(f"     Memory: {mem_mb:,.0f} MB ({percent_ram:.1f}% of 128GB)")
                    
                    if eta_hours > 0:
                        if eta_hours >= 1:
                            logger.info(f"     ETA: ~{eta_hours:.1f} hours")
                        else:
                            logger.info(f"     ETA: ~{eta_minutes:.0f} minutes")
                    
                    last_progress_time = current_time
                    last_progress_chunks = processed
            
            # Save index
            faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
            with open(IDS_MAPPING_PATH, 'wb') as f:
                pickle.dump(self.id_mapping, f)
            
            total_elapsed = time_module.time() - build_start_time
            elapsed_str = f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s"
            final_mem = process.memory_info().rss / (1024 * 1024)
            
            logger.info(f"")
            logger.info(f"✅ Embedding index created: {len(self.id_mapping):,} vectors")
            logger.info(f"   Total time: {elapsed_str}")
            logger.info(f"   Peak memory: {final_mem:,.0f} MB")
            logger.info(f"")
            
        except Exception as e:
            logger.error(f"Failed to build embedding index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.use_bm25_only = True
    
    def embedding_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """Search using embeddings - best for semantic queries"""
        if self.faiss_index is None or self.use_bm25_only:
            logger.warning("Embedding search not available, falling back to LIKE search")
            return self.bm25_search(query, top_k)
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k * 3)
            
            results = []
            cursor = self.conn.cursor()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                chunk_id = self.id_mapping.get(idx)
                if not chunk_id:
                    continue
                
                # Get full metadata
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
                        'score': float(1 / (1 + score))  # Convert distance to similarity
                    })
            
            elapsed = time.time() - start_time
            logger.info(f"Embedding search completed in {elapsed:.2f}s, found {len(results)} results")
            
            return results
        
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return self.bm25_search(query, top_k)
    
    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """Fast search using simple SQL LIKE - Fallback method"""
        start_time = time.time()
        
        try:
            cursor = self.conn.cursor()
            
            # Use simple LIKE search - fast fallback
            keywords = query.lower().split()
            
            # Build WHERE clause
            where_conditions = []
            params = []
            
            for keyword in keywords:
                where_conditions.append("(title LIKE ? OR text LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            sql = f"""
                SELECT 
                    id, text, title, page_id, url, date_modified,
                    wikidata_id, infoboxes, has_math, start_pos, end_pos
                FROM chunks
                WHERE {where_clause}
                LIMIT ?
            """
            
            params.append(top_k * 5)
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                # Calculate relevance score
                title_matches = sum(1 for kw in keywords if kw in row[2].lower())
                text_matches = sum(1 for kw in keywords if kw in row[1].lower())
                score = (title_matches * 2 + text_matches) / (len(keywords) * 3)
                
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
                    'score': score
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            elapsed = time.time() - start_time
            logger.info(f"LIKE search completed in {elapsed:.2f}s, found {len(results)} results")
            
            return results[:top_k * 5]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search(self, query: str, top_k: int = 20, method: str = "embedding") -> List[Dict[str, Any]]:
        """Unified search method
        
        Args:
            query: Search query
            top_k: Number of results
            method: "embedding" for semantic search, "like" for keyword search
        """
        start_time = time.time()
        
        if method == "embedding":
            results = self.embedding_search(query, RETRIEVAL_TOPK)
        else:
            results = self.bm25_search(query, RETRIEVAL_TOPK)
        
        # Rerank top results
        reranked_results = self.rerank_results(query, results, top_k)
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.2f}s: '{query}' → {len(reranked_results)} results")
        
        return reranked_results
    
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
    
    logger.info("Testing HybridRetriever with embedding search")
    
    # Initialize with embedding search (will build index on first run)
    retriever = HybridRetriever(use_bm25_only=False, build_embeddings=False)
    retriever.load_indexes()
    
    # Test search
    test_queries = [
        "ancient roman architecture",
        "quantum physics research",
        "renaissance art and culture"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=3, method="embedding")
        
        logger.info(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['title']} (score: {result.get('rerank_score', 0):.3f})")
            logger.info(f"   {result['text'][:150]}...")
    
    retriever.close()

