"""
Hybrid retrieval system for WikiTalk
"""
import sqlite3
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import json

from config import *


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.faiss_index = None
        self.id_mapping = {}
        self.conn = None
        
    def load_indexes(self):
        """Load SQLite and FAISS indexes"""
        print("Loading retrieval indexes...")
        
        # Load FAISS index
        if FAISS_INDEX_PATH.exists():
            self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(IDS_MAPPING_PATH, 'rb') as f:
                self.id_mapping = pickle.load(f)
        else:
            raise FileNotFoundError("FAISS index not found. Run data_processor.py first.")
        
        # Connect to SQLite
        self.conn = sqlite3.connect(SQLITE_DB_PATH)
        print("Indexes loaded successfully")
    
    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """BM25 search using SQLite FTS5"""
        cursor = self.conn.cursor()
        
        # FTS5 search with ranking
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
        
        return results
    
    def dense_search(self, query: str, top_k: int = RETRIEVAL_TOPK) -> List[Dict[str, Any]]:
        """Dense vector search using FAISS"""
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
    
    def hybrid_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Combine BM25 and dense search results"""
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
            # Normalize and combine scores
            bm25_score = result.get('bm25_score', 0)
            dense_score = result.get('dense_score', 0)
            
            # Simple linear combination
            result['combined_score'] = 0.6 * dense_score + 0.4 * bm25_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Rerank top results
        top_results = combined_results[:top_k * 2]  # Get more for reranking
        reranked_results = self.rerank_results(query, top_results, top_k)
        
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


if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.load_indexes()
    
    # Test search
    query = "World War I causes"
    results = retriever.hybrid_search(query, top_k=5)
    
    print(f"Search results for: {query}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Score: {result.get('combined_score', 0):.3f}")
        print(f"   Text: {result['text'][:200]}...")
    
    retriever.close()

