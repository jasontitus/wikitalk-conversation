#!/usr/bin/env python3
"""
Convert existing FAISS index to memory-optimized format

This script converts your existing 48GB FAISS index to a smaller format
WITHOUT re-generating embeddings - it just changes how the vectors are stored.
"""

import argparse
import faiss
import numpy as np
import pickle
import time
from pathlib import Path
from config import FAISS_INDEX_PATH, IDS_MAPPING_PATH

def convert_index(input_index_path, input_ids_path, output_index_path, output_ids_path, index_type):
    """Convert existing FAISS index to new format"""
    
    print(f"🔄 Converting {input_index_path} to {index_type.upper()} format...")
    
    # Load existing index
    print("📖 Loading existing FAISS index...")
    start_time = time.time()
    old_index = faiss.read_index(str(input_index_path))
    load_time = time.time() - start_time
    print(f"   ✓ Loaded in {load_time:.1f}s")
    
    # Load ID mapping
    print("📖 Loading ID mapping...")
    with open(input_ids_path, 'rb') as f:
        old_id_mapping = pickle.load(f)
    print(f"   ✓ Loaded {len(old_id_mapping):,} ID mappings")
    
    # Extract vectors from existing index
    print("🔍 Extracting vectors from existing index...")
    start_time = time.time()
    
    # Get total number of vectors
    n_vectors = old_index.ntotal
    print(f"   Total vectors: {n_vectors:,}")
    
    # Extract vectors in batches to avoid memory issues
    batch_size = 10000
    all_vectors = []
    
    for i in range(0, n_vectors, batch_size):
        end_idx = min(i + batch_size, n_vectors)
        batch_vectors = old_index.reconstruct_n(i, end_idx - i)
        all_vectors.append(batch_vectors)
        
        if i % 100000 == 0:
            print(f"   Extracted {i:,}/{n_vectors:,} vectors...")
    
    # Combine all vectors
    vectors = np.vstack(all_vectors).astype('float32')
    extract_time = time.time() - start_time
    print(f"   ✓ Extracted {len(vectors):,} vectors in {extract_time:.1f}s")
    
    # Create new index based on type
    print(f"🏗️  Creating new {index_type.upper()} index...")
    start_time = time.time()
    
    dimension = vectors.shape[1]
    
    if index_type == "ivfpq":
        # Product Quantization - good accuracy, much smaller memory (8-12GB)
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = 1000
        m = 64  # Number of sub-vectors (384/6 = 64, each sub-vector has 6 dimensions)
        bits = 8  # Bits per sub-vector (2^8 = 256 centroids)
        new_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
        print(f"   ✓ IVFPQ: 8-12GB memory, good accuracy (m={m}, bits={bits})")
        
    elif index_type == "ivfsq":
        # Scalar Quantization - good accuracy, smaller memory (12-16GB)
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = 1000
        new_index = faiss.IndexIVFScalarQuantizer(
            quantizer, dimension, nlist, faiss.ScalarQuantizer.QT_8bit
        )
        print(f"   ✓ IVFSQ: 12-16GB memory, good accuracy (8-bit quantization)")
        
    elif index_type == "hnsw":
        # HNSW - good accuracy, moderate memory (16-24GB), faster search
        M = 32  # Number of connections per node
        new_index = faiss.IndexHNSWFlat(dimension, M)
        print(f"   ✓ HNSW: 16-24GB memory, good accuracy, faster search (M={M})")
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Train the index (required for IVF-based indexes)
    if index_type in ["ivfpq", "ivfsq"]:
        print("🎓 Training index...")
        train_start = time.time()
        
        # Use a subset for training
        n_train = min(100000, len(vectors))
        train_vectors = vectors[:n_train]
        faiss.normalize_L2(train_vectors)
        new_index.train(train_vectors)
        
        train_time = time.time() - train_start
        print(f"   ✓ Trained on {n_train:,} vectors in {train_time:.1f}s")
    
    # Add vectors to new index
    print("➕ Adding vectors to new index...")
    add_start = time.time()
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # Add in batches to avoid memory issues
    batch_size = 50000
    for i in range(0, len(vectors), batch_size):
        end_idx = min(i + batch_size, len(vectors))
        batch = vectors[i:end_idx]
        new_index.add(batch)
        
        if i % 500000 == 0:
            print(f"   Added {i:,}/{len(vectors):,} vectors...")
    
    add_time = time.time() - add_start
    print(f"   ✓ Added all vectors in {add_time:.1f}s")
    
    # Save new index
    print("💾 Saving new index...")
    save_start = time.time()
    faiss.write_index(new_index, str(output_index_path))
    
    # Copy ID mapping (same as original)
    with open(output_ids_path, 'wb') as f:
        pickle.dump(old_id_mapping, f)
    
    save_time = time.time() - save_start
    print(f"   ✓ Saved in {save_time:.1f}s")
    
    # Report results
    total_time = time.time() - start_time
    old_size = input_index_path.stat().st_size / (1024**3)
    new_size = output_index_path.stat().st_size / (1024**3)
    savings = ((old_size - new_size) / old_size) * 100
    
    print(f"\n✅ Conversion completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Original size: {old_size:.1f} GB")
    print(f"   New size: {new_size:.1f} GB")
    print(f"   Memory savings: {savings:.1f}%")
    print(f"   New index: {output_index_path}")
    print(f"   New mapping: {output_ids_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert existing FAISS index to memory-optimized format")
    parser.add_argument(
        "--index-type", 
        choices=["ivfpq", "ivfsq", "hnsw"],
        default="ivfpq",
        help="Type of FAISS index to create (default: ivfpq - best balance)"
    )
    parser.add_argument(
        "--input-index",
        default=FAISS_INDEX_PATH,
        help="Path to input FAISS index (default: current index)"
    )
    parser.add_argument(
        "--input-ids",
        default=IDS_MAPPING_PATH,
        help="Path to input ID mapping (default: current mapping)"
    )
    
    args = parser.parse_args()
    
    # Create output paths
    index_suffix = f"_{args.index_type}"
    output_index_path = args.input_index.parent / f"faiss{index_suffix}.index"
    output_ids_path = args.input_ids.parent / f"ids{index_suffix}.bin"
    
    # Check if input files exist
    if not args.input_index.exists():
        print(f"❌ Input index not found: {args.input_index}")
        return
    
    if not args.input_ids.exists():
        print(f"❌ Input ID mapping not found: {args.input_ids}")
        return
    
    # Check if output files already exist
    if output_index_path.exists():
        print(f"❌ Output index already exists: {output_index_path}")
        print("   Delete it first if you want to recreate it")
        return
    
    if output_ids_path.exists():
        print(f"❌ Output ID mapping already exists: {output_ids_path}")
        print("   Delete it first if you want to recreate it")
        return
    
    # Run conversion
    convert_index(
        args.input_index,
        args.input_ids,
        output_index_path,
        output_ids_path,
        args.index_type
    )

if __name__ == "__main__":
    main()


