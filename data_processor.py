"""
Data processing and indexing for WikiTalk
"""
import sqlite3
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import fastparquet as fp
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

from config import *


def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def process_single_file_worker(file_path):
    """Standalone worker function for multiprocessing with ultra-conservative memory usage"""
    import gc
    try:
        # Read parquet file and process in very small batches to avoid memory issues
        df = fp.ParquetFile(str(file_path)).to_pandas()
        file_chunks = []
        
        # Process in very small batches to avoid memory overflow
        batch_size = 50  # Ultra-small batch size for memory efficiency
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            # Get a batch of rows
            end_idx = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:end_idx]
            
            for _, row in batch_df.iterrows():
                # Skip very short articles
                if len(row['text']) < 100:
                    continue
                
                # Create chunks using the same logic as the class method
                text = re.sub(r'\s+', ' ', row['text'].strip())
                chunks = []
                start = 0
                chunk_id = 0
                
                while start < len(text):
                    end = min(start + CHUNK_SIZE, len(text))
                    chunk_text = text[start:end]
                    
                    # Try to break at sentence boundaries
                    if end < len(text):
                        last_period = chunk_text.rfind('.')
                        last_newline = chunk_text.rfind('\n')
                        break_point = max(last_period, last_newline)
                        if break_point > start + CHUNK_SIZE // 2:
                            chunk_text = chunk_text[:break_point + 1]
                            end = start + break_point + 1
                    
                    # Create unique ID with timestamp to avoid conflicts
                    import time
                    unique_id = f"{row['page_id']}_{chunk_id}_{int(time.time() * 1000000)}"
                    
                    chunk = {
                        'id': unique_id,
                        'text': chunk_text,
                        'title': row['title'],
                        'page_id': row['page_id'],
                        'start_pos': start,
                        'end_pos': end,
                        'url': row['url'],
                        'date_modified': row['date_modified'],
                        'wikidata_id': row['wikidata_id'],
                        'infoboxes': row['infoboxes'],
                        'has_math': row['has_math']
                    }
                    
                    chunks.append(chunk)
                    start = max(start + CHUNK_SIZE - CHUNK_OVERLAP, end)
                    chunk_id += 1
                
                file_chunks.extend(chunks)
            
            # Clean up memory after each batch
            del batch_df
            gc.collect()
            
            # Additional cleanup every 2 batches
            if i % (batch_size * 2) == 0:
                gc.collect()
        
        # Clean up the main dataframe
        del df
        gc.collect()
        
        return file_chunks, str(file_path)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], str(file_path)


class DataProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = []
        self.id_mapping = {}
        
    def chunk_text(self, text: str, title: str, page_id: int) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + CHUNK_SIZE // 2:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            # Create unique ID with timestamp to avoid conflicts
            import time
            unique_id = f"{page_id}_{chunk_id}_{int(time.time() * 1000000)}"
            
            chunks.append({
                'id': unique_id,
                'text': chunk_text,
                'title': title,
                'page_id': page_id,
                'start_pos': start,
                'end_pos': end
            })
            
            start = max(start + CHUNK_SIZE - CHUNK_OVERLAP, end)
            chunk_id += 1
            
        return chunks
    
    def process_single_file(self, file_path):
        """Process a single parquet file and return chunks"""
        try:
            # Read parquet file
            df = fp.ParquetFile(str(file_path)).to_pandas()
            file_chunks = []
            
            # Process articles in batches for better memory usage
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                for _, row in batch_df.iterrows():
                    # Skip very short articles
                    if len(row['text']) < 100:
                        continue
                    
                    # Create chunks
                    article_chunks = self.chunk_text(
                        row['text'], 
                        row['title'], 
                        row['page_id']
                    )
                    
                    for chunk in article_chunks:
                        chunk['url'] = row['url']
                        chunk['date_modified'] = row['date_modified']
                        chunk['wikidata_id'] = row['wikidata_id']
                        chunk['infoboxes'] = row['infoboxes']
                        chunk['has_math'] = row['has_math']
                        
                        file_chunks.append(chunk)
            
            return file_chunks, str(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return [], str(file_path)

    def process_parquet_files(self):
        """Process all parquet files and create chunks using simple sequential processing"""
        print("Processing parquet files...")
        
        parquet_files = list(FINEWIKI_DIR.glob("*.parquet"))
        print(f"Found {len(parquet_files)} parquet files")
        
        all_chunks = []
        
        # Process files sequentially - simple approach
        for file_path in tqdm(parquet_files, desc="Processing files"):
            try:
                print(f"Processing {file_path.name}...")
                
                # Read parquet file
                df = fp.ParquetFile(str(file_path)).to_pandas()
                file_chunks = []
                
                for _, row in df.iterrows():
                    # Skip very short articles
                    if len(row['text']) < 100:
                        continue
                    
                    # Create chunks using the same logic as the class method
                    text = re.sub(r'\s+', ' ', row['text'].strip())
                    chunks = []
                    start = 0
                    chunk_id = 0
                    
                    while start < len(text):
                        end = min(start + CHUNK_SIZE, len(text))
                        chunk_text = text[start:end]
                        
                        # Try to break at sentence boundaries
                        if end < len(text):
                            last_period = chunk_text.rfind('.')
                            last_newline = chunk_text.rfind('\n')
                            break_point = max(last_period, last_newline)
                            if break_point > start + CHUNK_SIZE // 2:
                                chunk_text = chunk_text[:break_point + 1]
                                end = start + break_point + 1
                        
                        # Create unique ID with timestamp to avoid conflicts
                        import time
                        unique_id = f"{row['page_id']}_{chunk_id}_{int(time.time() * 1000000)}"
                        
                        chunk = {
                            'id': unique_id,
                            'text': chunk_text,
                            'title': row['title'],
                            'page_id': row['page_id'],
                            'start_pos': start,
                            'end_pos': end,
                            'url': row['url'],
                            'date_modified': row['date_modified'],
                            'wikidata_id': row['wikidata_id'],
                            'infoboxes': row['infoboxes'],
                            'has_math': row['has_math']
                        }
                        
                        chunks.append(chunk)
                        start = max(start + CHUNK_SIZE - CHUNK_OVERLAP, end)
                        chunk_id += 1
                    
                    file_chunks.extend(chunks)
                
                all_chunks.extend(file_chunks)
                print(f"✓ Processed {file_path.name}: {len(file_chunks)} chunks")
                
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
                continue
        
        self.chunks = all_chunks
        print(f"Created {len(self.chunks)} chunks from {len(parquet_files)} files")
    
    def create_sqlite_index(self):
        """Create SQLite FTS5 index for BM25 search"""
        print("Creating SQLite FTS5 index...")
        
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Create FTS5 virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                title,
                content='chunks',
                content_rowid='rowid'
            )
        """)
        
        # Create regular table for metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT,
                title TEXT,
                page_id INTEGER,
                url TEXT,
                date_modified TEXT,
                wikidata_id TEXT,
                infoboxes TEXT,
                has_math BOOLEAN,
                start_pos INTEGER,
                end_pos INTEGER
            )
        """)
        
        # Insert chunks
        for i, chunk in enumerate(tqdm(self.chunks, desc="Indexing chunks")):
            try:
                cursor.execute("""
                    INSERT INTO chunks (
                        id, text, title, page_id, url, date_modified, 
                        wikidata_id, infoboxes, has_math, start_pos, end_pos
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk['id'], chunk['text'], chunk['title'], chunk['page_id'],
                    chunk['url'], chunk['date_modified'], chunk['wikidata_id'],
                    chunk['infoboxes'], chunk['has_math'], chunk['start_pos'], chunk['end_pos']
                ))
                
                # Insert into FTS5
                cursor.execute("""
                    INSERT INTO chunks_fts (text, title) VALUES (?, ?)
                """, (chunk['text'], chunk['title']))
                
                # Store mapping for FAISS
                self.id_mapping[i] = chunk['id']
                
            except sqlite3.IntegrityError as e:
                print(f"Warning: Skipping duplicate chunk {chunk['id']}: {e}")
                continue
        
        conn.commit()
        conn.close()
        print("SQLite index created successfully")
    
    def create_faiss_index(self):
        """Create FAISS index for dense retrieval with memory-efficient streaming"""
        print("Creating FAISS index...")
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in self.chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Use smaller batches for memory efficiency
        batch_size = 256  # Smaller batch size for memory efficiency
        all_embeddings = []
        
        # Set number of threads for embedding generation
        import torch
        torch.set_num_threads(1)  # Use single thread to avoid memory issues
        
        # Configure sentence transformer for memory efficiency
        self.embedding_model.max_seq_length = 256  # Smaller max length
        
        # Process in smaller batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Use memory-efficient encoding
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                show_progress_bar=False,
                batch_size=32,  # Much smaller internal batch for memory efficiency
                convert_to_numpy=True,
                device='cpu',  # Force CPU to avoid GPU memory issues
                normalize_embeddings=True  # Normalize directly for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
            
            # Clean up memory after each batch
            import gc
            gc.collect()
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
        # No need to normalize again since we did it during encoding
        index.add(self.embeddings)
        
        # Save index and mapping
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(IDS_MAPPING_PATH, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        
        print("FAISS index created successfully")
    
    def load_chunks_from_sqlite(self):
        """Load chunks from existing SQLite database"""
        print("Loading chunks from SQLite database...")
        
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Get all chunks from the database
        cursor.execute("SELECT * FROM chunks ORDER BY rowid")
        columns = [description[0] for description in cursor.description]
        
        self.chunks = []
        for row in tqdm(cursor.fetchall(), desc="Loading chunks"):
            chunk = dict(zip(columns, row))
            self.chunks.append(chunk)
            # Build id mapping
            rowid = len(self.chunks) - 1
            self.id_mapping[rowid] = chunk['id']
        
        conn.close()
        print(f"Loaded {len(self.chunks)} chunks from database")
    
    def create_faiss_index_from_sqlite(self):
        """Create FAISS index directly from SQLite database without loading all chunks into memory"""
        print("Creating FAISS index from SQLite database...")
        
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Get total number of chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        print(f"Generating embeddings for {total_chunks} chunks...")
        
        # Use smaller batches for memory efficiency
        batch_size = 512  # Batch size for memory efficiency
        all_embeddings = []
        
        # Set number of threads for embedding generation
        import torch
        torch.set_num_threads(8)  # Use 8 threads for better performance
        
        # Configure sentence transformer for memory efficiency
        self.embedding_model.max_seq_length = 256  # Smaller max length
        
        # Process in smaller batches directly from SQLite
        for offset in tqdm(range(0, total_chunks, batch_size), desc="Generating embeddings"):
            # Get batch of texts from SQLite
            cursor.execute(
                "SELECT id, text FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
                (batch_size, offset)
            )
            rows = cursor.fetchall()
            
            batch_texts = [row[1] for row in rows]
            batch_ids = [row[0] for row in rows]
            
            # Use memory-efficient encoding
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                show_progress_bar=False,
                batch_size=64,  # Moderate internal batch for speed
                convert_to_numpy=True,
                device='cpu',  # Force CPU to avoid GPU memory issues
                normalize_embeddings=True  # Normalize directly for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
            
            # Build id mapping for this batch
            for i, chunk_id in enumerate(batch_ids):
                self.id_mapping[offset + i] = chunk_id
            
            # Clean up memory after each batch
            import gc
            gc.collect()
        
        conn.close()
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
        index.add(self.embeddings)
        
        # Save index and mapping
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(IDS_MAPPING_PATH, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        
        print("FAISS index created successfully")
    
    def process_all(self):
        """Process all data and create indexes"""
        print("Starting data processing...")
        
        # Check if SQLite database already exists
        if SQLITE_DB_PATH.exists():
            print("SQLite database found. Skipping chunk loading to save memory...")
        else:
            print("Processing parquet files...")
            # Process parquet files
            self.process_parquet_files()
            
            if not self.chunks:
                print("No chunks created. Exiting.")
                return
            
            # Create SQLite index
            self.create_sqlite_index()
        
        # Always create FAISS index (can be run independently)
        if not FAISS_INDEX_PATH.exists():
            print("Creating FAISS index...")
            if SQLITE_DB_PATH.exists():
                # Load directly from SQLite to save memory
                self.create_faiss_index_from_sqlite()
            else:
                # Use in-memory chunks if available
                self.create_faiss_index()
        else:
            print("FAISS index already exists. Skipping.")
        
        print("Data processing completed successfully!")


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    mp.set_start_method('spawn', force=True)
    
    processor = DataProcessor()
    processor.process_all()

