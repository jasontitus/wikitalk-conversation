"""
Test script for WikiTalk components
"""
import sys
import os
import logging
sys.path.append('.')

from data_processor import DataProcessor
from retriever import HybridRetriever
from llm_client import LLMClient, ConversationManager
from tts_client import TTSClient
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_processing():
    """Test data processing with a small sample"""
    logger.info("🧪 Testing data processing...")
    
    try:
        processor = DataProcessor()
        
        # Test chunking with a sample article
        sample_text = """
        World War I, also known as the Great War, was a global war that lasted from 1914 to 1918. 
        It involved many of the world's great powers, organized into two opposing alliances: 
        the Allies and the Central Powers. The war was triggered by the assassination of 
        Archduke Franz Ferdinand of Austria-Hungary in 1914.
        """
        
        chunks = processor.chunk_text(sample_text, "World War I", 12345)
        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(f"✓ First chunk: {chunks[0]['text'][:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data processing test failed: {e}")
        return False

def test_llm_client():
    """Test LLM client"""
    logger.info("🧪 Testing LLM client...")
    
    try:
        client = LLMClient()
        logger.info("✓ LLM Client initialized")
        
        # Test conversation manager
        manager = ConversationManager()
        session_id = "test_session"
        logger.info(f"✓ Conversation manager initialized")
        
        # Add some test exchanges
        manager.add_exchange(session_id, "Hello", "Hi there!")
        manager.add_exchange(session_id, "Tell me about history", "History is fascinating!")
        
        conversation = manager.load_conversation(session_id)
        logger.info(f"✓ Conversation manager working: {len(conversation['history'])} messages")
        
        # Try to test actual LLM connection (will likely fail without server)
        logger.info("\n📡 Testing LLM API connection...")
        try:
            logger.info(f"   Attempting connection to: {LLM_URL}")
            test_prompt = "Say 'Hello' in one word."
            response = client._call_llm(test_prompt, max_tokens=10)
            logger.info(f"✓ LLM API is working!")
            logger.info(f"   Response: {response}")
            return True
        except Exception as llm_error:
            logger.warning(f"⚠️ LLM API not available: {llm_error}")
            logger.info(f"\n💡 To use LLM features:")
            logger.info(f"   1. Download LM Studio from https://lmstudio.ai")
            logger.info(f"   2. Load a model (e.g., mistral, llama2)")
            logger.info(f"   3. Start the server (default: http://localhost:1234)")
            logger.info(f"\n   Current config points to: {LLM_URL}")
            return False
        
    except Exception as e:
        logger.error(f"❌ LLM client test failed: {e}")
        return False

def test_tts_client():
    """Test TTS client"""
    logger.info("🧪 Testing TTS client...")
    
    try:
        tts = TTSClient()
        
        # Test TTS availability
        if tts.use_piper:
            logger.info("✓ Piper TTS available")
        else:
            logger.info("⚠ Using fallback macOS 'say' command")
        
        # Test with a short phrase
        test_text = "Hello, this is a test."
        logger.info(f"✓ TTS client initialized")
        logger.info(f"  (Skipping audio output in test mode)")
        
        return True
    except Exception as e:
        logger.error(f"❌ TTS client test failed: {e}")
        return False

def test_retriever_setup():
    """Test retriever setup (without full index)"""
    logger.info("🧪 Testing retriever setup...")
    
    try:
        retriever = HybridRetriever()
        logger.info("✓ Retriever initialized")
        
        # Check if test database exists
        from pathlib import Path
        test_db = Path("data/test_docs.sqlite")
        if test_db.exists():
            logger.info(f"✓ Test database found: {test_db}")
            logger.info(f"  Size: {test_db.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            logger.info("⚠ Note: Test database not found (run test_simple_data.py)")
        
        full_db = Path("data/docs.sqlite")
        if full_db.exists():
            logger.info(f"✓ Full database found: {full_db}")
            logger.info(f"  Size: {full_db.stat().st_size / 1024 / 1024 / 1024:.1f} GB")
        
        return True
    except Exception as e:
        logger.warning(f"⚠ Retriever setup issue: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 WikiTalk Component Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Data Processing", test_data_processing),
        ("LLM Client", test_llm_client),
        ("TTS Client", test_tts_client),
        ("Retriever Setup", test_retriever_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Create test database: python test_simple_data.py")
    logger.info("2. Test retriever: python test_simple_retriever.py")
    logger.info("3. Start LLM server (LM Studio or llama.cpp)")
    logger.info("4. Run WikiTalk: python wikitalk.py")

if __name__ == "__main__":
    main()

