#!/usr/bin/env python3
"""
LM Studio Diagnostics
Helps troubleshoot LM Studio server connection issues
"""
import socket
import requests
import json
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_port_open(host, port, timeout=2):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking port: {e}")
        return False

def test_basic_connectivity(url):
    """Test basic HTTP connectivity"""
    logger.info(f"\n1️⃣ Testing basic connectivity to {url}")
    try:
        response = requests.get(url, timeout=5)
        logger.info(f"✓ Server is responding with status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ Connection refused: {e}")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"✗ Request timed out: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False

def test_chat_api(url):
    """Test the chat completions API"""
    logger.info(f"\n2️⃣ Testing chat API endpoint")
    
    payload = {
        "model": "any-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    try:
        logger.info(f"   Sending request to {url}")
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        logger.info(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ API is working!")
            if 'choices' in data and data['choices']:
                logger.info(f"   Response: {data['choices'][0]['message']['content']}")
            return True
        elif response.status_code == 400:
            logger.warning(f"✗ Bad request (model might not be loaded)")
            logger.info(f"   Error: {response.text}")
            return False
        elif response.status_code == 500:
            logger.error(f"✗ Server error")
            logger.info(f"   Error: {response.text}")
            return False
        else:
            logger.error(f"✗ Unexpected status code: {response.status_code}")
            logger.info(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"✗ Request timed out (server might be loading)")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ Connection refused: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def test_models_endpoint(host, port):
    """Test if we can get list of available models"""
    logger.info(f"\n3️⃣ Testing models endpoint")
    
    url = f"http://{host}:{port}/v1/models"
    
    try:
        logger.info(f"   Requesting models from {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Models endpoint working!")
            if 'data' in data:
                logger.info(f"   Available models: {len(data['data'])}")
                for model in data['data'][:3]:
                    logger.info(f"     - {model.get('id', 'Unknown')}")
            return True
        else:
            logger.warning(f"⚠ Models endpoint returned: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Cannot access models: {e}")
        return False

def run_diagnostics():
    """Run all diagnostics"""
    host = "localhost"
    port = 1234
    base_url = f"http://{host}:{port}"
    chat_url = f"{base_url}/v1/chat/completions"
    
    logger.info("🔍 LM Studio Diagnostics")
    logger.info("=" * 60)
    
    # Check port
    logger.info(f"\n🔎 Checking if port {port} is open...")
    if check_port_open(host, port):
        logger.info(f"✓ Port {port} is open")
    else:
        logger.error(f"✗ Port {port} is NOT open")
        logger.info("\n⚠️ LM Studio might not be running!")
        logger.info("   1. Open LM Studio app")
        logger.info("   2. Download a model (if not done)")
        logger.info("   3. Click 'Start Server'")
        logger.info("   4. Wait for 'Server Started' message")
        return False
    
    # Test basic connectivity
    logger.info(f"\n🔎 Testing basic connectivity...")
    if not test_basic_connectivity(base_url):
        logger.error("\n✗ Server is not responding to HTTP requests")
        logger.info("   Possible issues:")
        logger.info("   - Server is still starting up")
        logger.info("   - Firewall blocking connection")
        logger.info("   - Wrong port number")
        return False
    
    # Test models endpoint
    logger.info(f"\n🔎 Checking for loaded models...")
    if not test_models_endpoint(host, port):
        logger.warning("\n⚠️ Cannot access models endpoint")
        logger.info("   - Model might still be loading")
        logger.info("   - Try again in a few seconds")
    
    # Test chat API
    logger.info(f"\n🔎 Testing chat API...")
    if test_chat_api(chat_url):
        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS! LM Studio is ready to use")
        logger.info("\nYou can now run:")
        logger.info("  python test_llm_only.py")
        logger.info("  python wikitalk.py")
        return True
    else:
        logger.error("\n✗ Chat API is not responding correctly")
        logger.info("\n💡 Troubleshooting steps:")
        logger.info("   1. Make sure a model is downloaded in LM Studio")
        logger.info("   2. Click 'Start Server' button")
        logger.info("   3. Wait 30-60 seconds for server to fully start")
        logger.info("   4. Check LM Studio logs for errors")
        logger.info("   5. Try a different model if current one fails")
        logger.info("\n📝 LM Studio Models to try:")
        logger.info("   - Mistral 7B (recommended, smallest)")
        logger.info("   - Llama 2 7B")
        logger.info("   - Neural Chat 7B")
        return False

def show_lm_studio_instructions():
    """Show detailed LM Studio setup instructions"""
    logger.info("\n" + "=" * 60)
    logger.info("📖 LM Studio Setup Instructions")
    logger.info("=" * 60)
    
    logger.info("\n1. Download LM Studio")
    logger.info("   - Visit: https://lmstudio.ai")
    logger.info("   - Download macOS version")
    logger.info("   - Install to Applications")
    
    logger.info("\n2. Download a Model")
    logger.info("   - Open LM Studio app")
    logger.info("   - Click search icon on left")
    logger.info("   - Search for 'mistral' or 'neural-chat'")
    logger.info("   - Click download button (⬇️)")
    logger.info("   - Wait for download to complete (~5-10 minutes)")
    
    logger.info("\n3. Start the Server")
    logger.info("   - Look for 'Local Server' section")
    logger.info("   - Click 'Start Server'")
    logger.info("   - Wait for: 'Server started at http://localhost:1234'")
    logger.info("   - Keep LM Studio app open")
    
    logger.info("\n4. Test Connection")
    logger.info("   - Run: python diagnose_lm_studio.py")
    logger.info("   - Should show: ✅ SUCCESS!")
    
    logger.info("\n5. Run WikiTalk")
    logger.info("   - Run: python wikitalk.py")
    logger.info("   - Or: python test_wikitalk.py")

if __name__ == "__main__":
    logger.info("\n")
    success = run_diagnostics()
    
    if not success:
        show_lm_studio_instructions()
    
    logger.info("\n" + "=" * 60)
