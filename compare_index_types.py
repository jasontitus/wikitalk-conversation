#!/usr/bin/env python3
"""
Quick comparison of FAISS index types without rebuilding

This script shows the memory and accuracy tradeoffs of different index types
without requiring a full rebuild.
"""

def print_comparison():
    """Print a comparison table of different FAISS index types"""
    
    print("🔍 FAISS Index Type Comparison")
    print("=" * 80)
    print()
    
    # Index type data
    index_types = [
        {
            'name': 'IVFFlat (Original)',
            'memory_gb': 48,
            'accuracy': '100%',
            'search_speed': 'Standard',
            'build_time': '1-2 hours',
            'description': 'Exact vectors, highest accuracy',
            'best_for': 'Maximum accuracy, plenty of memory'
        },
        {
            'name': 'IVFPQ (Product Quantization)',
            'memory_gb': 8,
            'accuracy': '95-98%',
            'search_speed': 'Slightly slower',
            'build_time': '1-2 hours',
            'description': 'Compressed vectors, good accuracy',
            'best_for': 'Most users - best balance'
        },
        {
            'name': 'IVFSQ (Scalar Quantization)',
            'memory_gb': 12,
            'accuracy': '98-99%',
            'search_speed': 'Similar to original',
            'build_time': '1-2 hours',
            'description': '8-bit quantization, high accuracy',
            'best_for': 'High accuracy with moderate savings'
        },
        {
            'name': 'HNSW (Hierarchical)',
            'memory_gb': 16,
            'accuracy': '95-98%',
            'search_speed': 'Faster than original',
            'build_time': '1-2 hours',
            'description': 'Graph-based search, fast queries',
            'best_for': 'Fast search, moderate memory'
        }
    ]
    
    # Print comparison table
    print(f"{'Index Type':<20} {'Memory':<8} {'Accuracy':<10} {'Speed':<15} {'Best For'}")
    print("-" * 80)
    
    for idx in index_types:
        print(f"{idx['name']:<20} {idx['memory_gb']:<8}GB {idx['accuracy']:<10} {idx['search_speed']:<15} {idx['best_for']}")
    
    print()
    print("📊 Memory Savings vs Original:")
    print("-" * 40)
    for idx in index_types[1:]:  # Skip original
        savings = ((48 - idx['memory_gb']) / 48) * 100
        print(f"{idx['name']:<20} {savings:>6.1f}% smaller")
    
    print()
    print("🎯 Recommendations:")
    print("-" * 40)
    print("• Most users: IVFPQ (8GB, 95-98% accuracy)")
    print("• High accuracy: IVFSQ (12GB, 98-99% accuracy)")  
    print("• Fast search: HNSW (16GB, faster queries)")
    print("• Maximum accuracy: IVFFlat (48GB, 100% accuracy)")
    
    print()
    print("🚀 How to build different index types:")
    print("-" * 50)
    print("python build_embeddings_memory_optimized.py --index-type ivfpq")
    print("python build_embeddings_memory_optimized.py --index-type ivfsq")
    print("python build_embeddings_memory_optimized.py --index-type hnsw")
    print("python build_embeddings_memory_optimized.py --index-type ivfflat")

if __name__ == "__main__":
    print_comparison()


