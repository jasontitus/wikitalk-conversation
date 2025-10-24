"""
Setup script for WikiTalk
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Python requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_piper():
    """Check if Piper TTS is available"""
    print("Checking Piper TTS...")
    
    # Check if piper is in PATH
    try:
        result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Piper TTS found in PATH")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠ Piper TTS not found. You can install it from:")
    print("  https://github.com/rhasspy/piper")
    print("  Or use the fallback macOS 'say' command")
    return False

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    
    directories = [
        "data",
        "data/conversations",
        "voices"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up WikiTalk...")
    print("="*50)
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return False
    
    # Check Piper TTS
    check_piper()
    
    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Process Wikipedia data: python data_processor.py")
    print("2. Start LM Studio or llama.cpp server")
    print("3. Run WikiTalk: python wikitalk.py")
    print("\nFor TTS, download Piper voices to ./voices/ directory")
    
    return True

if __name__ == "__main__":
    main()

