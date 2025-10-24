# 🎮 GPU Setup Guide for Windows with NVIDIA Cards

## Problem: Default to CPU on Windows

When you launch WikiTalk on Windows with an NVIDIA GPU, it may default to CPU processing instead of using your GPU. This significantly slows down embedding generation (6-12 hours vs 1-3 hours).

## Solution: Enable GPU Support

### Step 1: Verify GPU is Available

Before proceeding with any fixes, check if your system can detect your GPU:

```bash
time python verify_gpu.py
```

This will show:
- ✓ PyTorch version
- ✓ CUDA status
- ✓ GPU devices found
- ✓ Sentence-Transformers GPU compatibility
- ✓ FAISS GPU support

**Expected output if GPU is working:**
```
✅ GPU is properly configured and ready to use!
   Your embedding builds will be significantly faster:
   - GPU (NVIDIA RTX 4090): ~1-2 hours
   - GPU (NVIDIA RTX 4070): ~2-3 hours
```

---

## If GPU Detection Fails

### Check 1: NVIDIA Driver Installation

```bash
nvidia-smi
```

**If command not found or error:**
- Download NVIDIA driver: https://www.nvidia.com/Download/driverDetails.aspx
- Select your GPU model and Windows version
- Install and restart your computer

**Expected output:**
```
NVIDIA-SMI 555.xx.xx    Driver Version: 555.xx.xx
CUDA Version: 12.5
...
+--------+---------+
| GPU Name: NVIDIA GeForce RTX 4070
| Memory: 12215MiB
+--------+---------+
```

### Check 2: CUDA Toolkit Installation

If `nvidia-smi` shows CUDA Version but PyTorch doesn't detect it:

1. **Download CUDA Toolkit:**
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 or 12 (match your driver)
   - Download and run installer

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```

3. **Reinstall PyTorch with CUDA support:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   Or for CUDA 12:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Check 3: cuDNN Installation (Optional but Recommended)

For faster operations, especially with FAISS:

1. Download cuDNN: https://developer.nvidia.com/cudnn
2. Extract and add to Python site-packages:
   ```bash
   # Copy cudnn files to your CUDA directory
   # Usually: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
   ```

---

## Verify Fix Works

After any installation changes, run the verification again:

```bash
time python verify_gpu.py
```

Now you should see GPU detection working:
```
✓ CUDA available: True
✓ CUDA version: 12.1
✓ cuDNN version: 8902
✓ Number of GPUs: 1
   GPU 0:
      Name: NVIDIA GeForce RTX 4070
      Total Memory: 12.0 GB
```

---

## Code Changes Made

### 1. **retriever.py** - GPU Device Detection

The `retriever.py` file now automatically detects and uses GPU:

```python
import torch

# Detect and set device for GPU acceleration
def get_device():
    """Get the best available device for inference"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("⚠ GPU not detected, using CPU (slower)")
    return device

DEVICE = get_device()

# Models now explicitly use GPU:
self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
```

### 2. **verify_gpu.py** - New Diagnostic Tool

Created a comprehensive GPU verification script that checks:
- PyTorch installation and version
- CUDA availability and version
- GPU devices and memory
- Sentence-Transformers GPU support
- FAISS GPU support
- Step-by-step troubleshooting hints

---

## Quick Fix Summary

### If your GPU is installed but not detected:

**Windows Command Prompt (Admin):**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
time python verify_gpu.py
```

### If CUDA is not installed:

1. Check driver: `nvidia-smi`
2. If no output → Install NVIDIA driver
3. Download CUDA Toolkit from nvidia.com
4. Reinstall PyTorch with CUDA support (see above)

---

## Performance Impact

| Configuration | Embedding Build Time |
|---|---|
| **GPU (RTX 4090)** | 1-2 hours ⚡ |
| **GPU (RTX 4070)** | 2-3 hours ⚡ |
| **GPU (RTX 3080)** | 3-4 hours ⚡ |
| **CPU only** | 6-12 hours 🐌 |

---

## Troubleshooting

### Problem: "CUDA out of memory"
- Reduce `BATCH_SIZE` in `config.py` from 2000 to 1000
- Or reduce `embedding_batch_size` in `retriever.py` from 1024 to 512

### Problem: "No CUDA-capable device"
- Run `nvidia-smi` in command prompt
- If command not found, NVIDIA drivers aren't installed

### Problem: PyTorch installed but CUDA not detected
- Wrong PyTorch version for your CUDA
- Uninstall and reinstall with matching CUDA version

### Problem: Slow performance even with GPU detected
- Check GPU utilization: Open Task Manager → GPU tab
- If 0%, likely FAISS not using GPU (still fast via CPU FAISS)

---

## Testing GPU Performance

To verify GPU is actually being used during embedding generation:

**Open Task Manager:**
1. Start embedding build: `time python build_embeddings.py`
2. Open Task Manager → GPU tab
3. Watch "GPU" utilization (should be 80-100% if working)

**Command line monitoring:**
```bash
nvidia-smi -l 1
```
This shows GPU usage every 1 second. Should show high utilization during embedding generation.

---

## Need Help?

If issues persist:

1. Check NVIDIA driver: `nvidia-smi`
2. Run diagnostics: `time python verify_gpu.py`
3. Check PyTorch docs: https://pytorch.org/get-started/locally/
4. Check FAISS GPU docs: https://github.com/facebookresearch/faiss/wiki/Faiss-on-GPU
