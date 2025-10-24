#!/usr/bin/env python3
"""
Simple test data processor - creates a basic searchable database without FTS5
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import fastparquet as fp
from tqdm import tqdm
import re

# Test database paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FINEWIKI_DIR = BASE_DIR / "finewiki" / "data" / "enwiki"
TEST_DB_PATH = DATA_DIR / "test_docs.sqlite"

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def process_test_data():
    """Process just 3 parquet files for testing - simple approach"""
    print("🧪 Processing test data (3 files only - simple database)...")
    
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
            
            for idx, (_, row) in enumerate(df.iterrows()):
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
                    
                    # Create unique ID with simple counter
                    unique_id = f"{row['page_id']}_{chunk_id}"
                    
                    chunk = {
                        'id': unique_id,
                        'text': chunk_text,
                        'title': row['title'],
                        'page_id': row['page_id'],
                    }
                    
                    chunks.append(chunk)
                    start = max(start + CHUNK_SIZE - CHUNK_OVERLAP, end)
                    chunk_id += 1
                
                file_chunks.extend(chunks)
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1} articles...")
            
            all_chunks.extend(file_chunks)
            print(f"✓ Processed {file_path.name}: {len(file_chunks)} chunks")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            continue
    
    print(f"\n✅ Created {len(all_chunks)} total chunks")
    
    # Create SQLite database - SIMPLE, no FTS5
    print("\n📦 Creating simple SQLite database...")
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    # Create simple table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            title TEXT,
            page_id INTEGER
        )
    """)
    
    # Add simple index on title for faster searches
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON chunks(title)")
    
    # Insert chunks
    for chunk in tqdm(all_chunks, desc="Inserting chunks"):
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO chunks (id, text, title, page_id)
                VALUES (?, ?, ?, ?)
            """, (chunk['id'], chunk['text'], chunk['title'], chunk['page_id']))
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    
    print(f"✅ Simple database created: {TEST_DB_PATH}")
    print(f"   Total chunks: {len(all_chunks):,}")
    return len(all_chunks)

if __name__ == "__main__":
    process_test_data()
