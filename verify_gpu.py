#!/usr/bin/env python3
"""
GPU Verification Script for WikiTalk
Helps diagnose GPU availability and configuration issues
"""

import sys
from pathlib import Path

def check_gpu():
    """Check GPU availability and configuration"""
    print("=" * 70)
    print("🔧 WikiTalk GPU Verification")
    print("=" * 70)
    print()
    
    # Check PyTorch
    print("1️⃣  PyTorch Installation:")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False
    
    # Check CUDA availability
    print()
    print("2️⃣  CUDA Status:")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✓ CUDA available: True")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print(f"   ✗ CUDA available: False")
        print()
        print("   ⚠  TROUBLESHOOTING:")
        print("   - Verify NVIDIA driver is installed")
        print("   - Run: nvidia-smi")
        print("   - If no output, driver may not be installed")
    
    # Check GPU devices
    print()
    print("3️⃣  GPU Devices:")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"   ✓ Number of GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Capability: {torch.cuda.get_device_capability(i)}")
            
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            print(f"      Total Memory: {total_memory:.1f} GB")
            
            # Get current memory usage
            try:
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"      Memory Allocated: {allocated:.1f} GB")
                print(f"      Memory Reserved: {reserved:.1f} GB")
            except:
                pass
    else:
        print("   ✗ No GPU devices found")
    
    # Check sentence-transformers
    print()
    print("4️⃣  Sentence-Transformers:")
    try:
        from sentence_transformers import SentenceTransformer
        print(f"   ✓ Sentence-Transformers installed")
        
        # Try to load model
        print(f"   ℹ  Testing model loading...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            device = "cuda" if cuda_available else "cpu"
            model = SentenceTransformer(model_name, device=device)
            print(f"   ✓ Model loaded on {device}")
            
            # Test encoding
            test_text = ["This is a test sentence"]
            embeddings = model.encode(test_text)
            print(f"   ✓ Model encoding works")
            print(f"      Embedding shape: {embeddings.shape}")
            
        except Exception as e:
            print(f"   ✗ Model loading failed: {e}")
    except ImportError:
        print("   ✗ Sentence-Transformers not installed")
    
    # Check FAISS
    print()
    print("5️⃣  FAISS:")
    try:
        import faiss
        print(f"   ✓ FAISS installed")
        
        if cuda_available:
            # Check if FAISS GPU support is available
            try:
                res = faiss.StandardGpuResources()
                print(f"   ✓ FAISS GPU support available")
            except:
                print(f"   ⚠  FAISS GPU support not available (CPU mode will be used)")
        else:
            print(f"   ℹ  FAISS will use CPU")
    except ImportError:
        print("   ✗ FAISS not installed")
    
    # Summary
    print()
    print("=" * 70)
    if cuda_available:
        print("✅ GPU is properly configured and ready to use!")
        print()
        print("   Your embedding builds will be significantly faster:")
        print("   - GPU (NVIDIA RTX 4090): ~1-2 hours")
        print("   - GPU (NVIDIA RTX 4070): ~2-3 hours")
        print("   - CPU only: ~6-12 hours")
    else:
        print("⚠  GPU is not available. System will use CPU (slower)")
        print()
        print("   On Windows with NVIDIA GPU:")
        print("   1. Install NVIDIA driver: https://www.nvidia.com/Download/driverDetails.aspx")
        print("   2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("   3. Install cuDNN: https://developer.nvidia.com/cudnn")
        print("   4. Reinstall PyTorch with CUDA support:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 70)
    print()
    
    return cuda_available

if __name__ == "__main__":
    cuda_available = check_gpu()
    sys.exit(0 if cuda_available else 1)
