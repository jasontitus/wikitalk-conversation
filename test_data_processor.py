#!/usr/bin/env python3
"""
Test data processor - processes only a few parquet files for quick testing
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
import os
import gc

# Test database paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FINEWIKI_DIR = BASE_DIR / "finewiki" / "data" / "enwiki"
TEST_DB_PATH = DATA_DIR / "test_docs.sqlite"

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

def process_test_data():
    """Process just 3 parquet files for testing"""
    print("🧪 Processing test data (3 files only)...")
    
    # Get only the first 3 parquet files
    parquet_files = sorted(list(FINEWIKI_DIR.glob("*.parquet")))[:3]
    print(f"Processing {len(parquet_files)} files: {[f.name for f in parquet_files]}")
    
    all_chunks = []
    
    # Process files sequentially
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            print(f"\nProcessing {file_path.name}...")
            
            # Read parquet file
            df = fp.ParquetFile(str(file_path)).to_pandas()
            file_chunks = []
            
            for _, row in df.iterrows():
                # Skip very short articles
                if len(row['text']) < 100:
                    continue
                
                # Create chunks
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
                    
                    # Create unique ID
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
    
    print(f"\n✅ Created {len(all_chunks)} total chunks")
    
    # Create SQLite database
    print("\n📦 Creating SQLite database...")
    conn = sqlite3.connect(TEST_DB_PATH)
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
    id_mapping = {}
    for i, chunk in enumerate(tqdm(all_chunks, desc="Indexing chunks")):
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
            
            # Store mapping
            id_mapping[i] = chunk['id']
            
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    
    print(f"✅ SQLite test database created: {TEST_DB_PATH}")
    print(f"   Total chunks indexed: {len(all_chunks):,}")
    return len(all_chunks)

if __name__ == "__main__":
    process_test_data()
