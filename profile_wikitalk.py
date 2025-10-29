"""
Performance profiling script for WikiTalk - identifies bottlenecks
"""
import time
import sys
from typing import Dict, Any, List
import logging

from retriever import HybridRetriever
from llm_client import LLMClient, ConversationManager

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

class PerformanceProfiler:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, phase: str):
        """Start timing a phase"""
        self.start_times[phase] = time.time()
    
    def end(self, phase: str) -> float:
        """End timing a phase and return elapsed time"""
        if phase not in self.start_times:
            return 0
        
        elapsed = time.time() - self.start_times[phase]
        
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(elapsed)
        
        return elapsed
    
    def report(self):
        """Print timing report"""
        print("\n" + "="*70)
        print("📊 PERFORMANCE PROFILING REPORT")
        print("="*70)
        
        total_time = sum(sum(times) for times in self.timings.values()) / len(next(iter(self.timings.values()))) if self.timings else 0
        
        for phase in sorted(self.timings.keys()):
            times = self.timings[phase]
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            pct = (avg / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{phase}:")
            print(f"  Average: {avg:>7.2f}s ({pct:>5.1f}%)")
            print(f"  Min:     {min_t:>7.2f}s")
            print(f"  Max:     {max_t:>7.2f}s")
            print(f"  Samples: {len(times)}")

def profile_query_processing(queries: List[str], num_runs: int = 2):
    """Profile query processing pipeline with granular timing"""
    profiler = PerformanceProfiler()
    
    print("\n🚀 Initializing WikiTalk components...")
    
    # Initialize components
    profiler.start("initialization")
    retriever = HybridRetriever(use_bm25_only=False)
    retriever.load_indexes()
    llm_client = LLMClient()
    conv_manager = ConversationManager()
    profiler.end("initialization")
    
    print(f"✓ Initialization complete")
    
    # GPU Acceleration for Apple Silicon
    print("\n🎮 Checking FAISS acceleration...")
    try:
        import faiss
        if retriever.faiss_index is not None:
            print(f"   FAISS Index Type: {type(retriever.faiss_index).__name__}")
            print(f"   Index Size: {retriever.faiss_index.ntotal:,} vectors")
            print(f"   ℹ️  CPU-optimized search (GPU support requires Metal-enabled build)")
    except Exception as e:
        print(f"⚠️  FAISS check error: {e}")
    
    # Warmup phase - cache the index and system
    print("\n🔥 Warming up FAISS index (3 queries)...")
    warmup_queries = [
        "ancient Rome",
        "quantum mechanics", 
        "world history"
    ]
    
    for i, warmup_q in enumerate(warmup_queries, 1):
        print(f"  Warmup {i}/3: '{warmup_q}'...", end=" ", flush=True)
        try:
            _ = retriever.search(warmup_q, top_k=5, method="embedding")
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")
    
    print("\n✅ Index warmed up and cached in memory/GPU")
    print("=" * 70)
    
    print(f"\n🧪 Running {num_runs} benchmark runs with {len(queries)} queries each...")
    print("="*70)
    
    session_id = "profile_session"
    conversation_data = conv_manager.load_conversation(session_id)
    history = conversation_data.get("history", [])
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        
        for query in queries:
            print(f"  Query: '{query}'")
            
            # Phase 1: Query rewrite
            profiler.start("query_rewrite")
            rewritten = llm_client.query_rewrite(query, history)
            elapsed = profiler.end("query_rewrite")
            print(f"    ✓ Query rewrite: {elapsed:.2f}s")
            
            # Phase 2: Semantic search with granular timing
            print(f"    🔍 Semantic search breakdown:")
            
            profiler.start("embedding_generation")
            from sentence_transformers import SentenceTransformer
            search_start = time.time()
            
            # Manually time embedding generation
            query_embedding = llm_client.__class__.__bases__[0]  # Get parent class
            import numpy as np
            import faiss
            
            embedding_model = retriever.embedding_model
            query_embedding = embedding_model.encode([rewritten], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            elapsed_embedding = time.time() - search_start
            profiler.timings.setdefault("embedding_generation", []).append(elapsed_embedding)
            print(f"      → Embedding generation: {elapsed_embedding:.2f}s")
            
            # Time FAISS search
            faiss_start = time.time()
            from config import RETRIEVAL_TOPK
            scores, indices = retriever.faiss_index.search(query_embedding, RETRIEVAL_TOPK * 2)
            elapsed_faiss = time.time() - faiss_start
            profiler.timings.setdefault("faiss_search", []).append(elapsed_faiss)
            print(f"      → FAISS search: {elapsed_faiss:.2f}s")
            
            # Time database batch lookup
            db_start = time.time()
            chunk_ids_with_scores = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk_id = retriever.id_mapping.get(idx)
                if chunk_id:
                    chunk_ids_with_scores.append((chunk_id, score))
            
            if chunk_ids_with_scores:
                cursor = retriever.conn.cursor()
                chunk_ids = [cid for cid, _ in chunk_ids_with_scores]
                score_map = {cid: score for cid, score in chunk_ids_with_scores}
                
                placeholders = ','.join('?' * len(chunk_ids))
                cursor.execute(f"""
                    SELECT id, text, title, page_id, url, date_modified,
                           wikidata_id, infoboxes, has_math, start_pos, end_pos
                    FROM chunks WHERE id IN ({placeholders})
                """, chunk_ids)
                
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    chunk_id = row[0]
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
                        'score': float(1 / (1 + score_map[chunk_id]))
                    })
            
            elapsed_db = time.time() - db_start
            profiler.timings.setdefault("database_batch_lookup", []).append(elapsed_db)
            print(f"      → Database batch lookup: {elapsed_db:.2f}s")
            
            total_search = time.time() - search_start
            profiler.timings.setdefault("semantic_search", []).append(total_search)
            print(f"    ✓ Total semantic search: {total_search:.2f}s ({len(results)} results)")
            
            # Phase 3: Response generation with LLM timing
            print(f"    💬 Response generation breakdown:")
            
            gen_start = time.time()
            llm_call_start = time.time()
            response = llm_client.generate_response(rewritten, results[:3], history)
            llm_call_elapsed = time.time() - llm_call_start
            
            profiler.timings.setdefault("llm_response_generation", []).append(llm_call_elapsed)
            print(f"      → LLM API call: {llm_call_elapsed:.2f}s")
            
            gen_elapsed = time.time() - gen_start
            profiler.timings.setdefault("response_generation", []).append(gen_elapsed)
            print(f"    ✓ Total response generation: {gen_elapsed:.2f}s")
            
            # Phase 4: Conversation save
            profiler.start("conversation_save")
            conv_manager.add_exchange(session_id, query, response)
            elapsed = profiler.end("conversation_save")
            print(f"    ✓ Conversation save: {elapsed:.2f}s")
    
    # Print summary
    profiler.report()
    
    # Print granular breakdown
    print("\n" + "="*70)
    print("🔍 GRANULAR BREAKDOWN (Semantic Search Components)")
    print("="*70)
    
    if "embedding_generation" in profiler.timings:
        avg_emb = sum(profiler.timings["embedding_generation"]) / len(profiler.timings["embedding_generation"])
        print(f"\n⚡ Embedding Generation: {avg_emb:.2f}s average")
        print(f"   → sentence_transformers.encode([query])")
    
    if "faiss_search" in profiler.timings:
        avg_faiss = sum(profiler.timings["faiss_search"]) / len(profiler.timings["faiss_search"])
        print(f"\n⚡ FAISS Search: {avg_faiss:.2f}s average")
        from config import RETRIEVAL_TOPK
        print(f"   → faiss_index.search(query_embedding, {RETRIEVAL_TOPK * 2})")
    
    if "database_batch_lookup" in profiler.timings:
        avg_db = sum(profiler.timings["database_batch_lookup"]) / len(profiler.timings["database_batch_lookup"])
        print(f"\n⚡ Database Batch Lookup: {avg_db:.2f}s average")
        print(f"   → Single SQL query with IN clause for 40 results")
    
    print("\n" + "="*70)
    print("🔍 GRANULAR BREAKDOWN (Response Generation)")
    print("="*70)
    
    if "llm_response_generation" in profiler.timings:
        avg_llm = sum(profiler.timings["llm_response_generation"]) / len(profiler.timings["llm_response_generation"])
        print(f"\n⚡ LLM API Call: {avg_llm:.2f}s average")
        print(f"   → LM Studio LLM inference with sources")
    
    retriever.close()

if __name__ == "__main__":
    test_queries = [
        "What was the French Revolution?",
        "Tell me about quantum physics",
        "Who was Napoleon Bonaparte?",
    ]
    
    print("\n" + "="*70)
    print("🔧 WIKITALK GRANULAR PERFORMANCE PROFILER")
    print("="*70)
    print("\nThis tool measures individual component times for detailed analysis.")
    print("Look for which sub-component is the actual bottleneck.\n")
    
    try:
        profile_query_processing(test_queries, num_runs=1)
    except KeyboardInterrupt:
        print("\n\n👋 Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
