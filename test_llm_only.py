#!/usr/bin/env python3
"""
Test just the LLM client and conversation manager
"""
import logging
import sys
sys.path.append('.')

from llm_client import LLMClient, ConversationManager
from config import LLM_URL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_llm_client():
    """Test LLM client connection and conversation management"""
    logger.info("🚀 WikiTalk LLM Client Test")
    logger.info("=" * 60)
    
    # Initialize LLM client
    logger.info("\n1️⃣ Initializing LLM Client...")
    client = LLMClient()
    
    # Initialize conversation manager
    logger.info("\n2️⃣ Initializing Conversation Manager...")
    manager = ConversationManager()
    
    # Test conversation management
    logger.info("\n3️⃣ Testing Conversation Management...")
    session_id = "demo_session"
    
    manager.add_exchange(
        session_id,
        "Tell me about the Roman Empire",
        "The Roman Empire was one of the greatest civilizations in history..."
    )
    
    manager.add_exchange(
        session_id,
        "What was their government like?",
        "The Roman Empire had a complex government structure..."
    )
    
    conversation = manager.load_conversation(session_id)
    logger.info(f"✓ Conversation has {len(conversation['history'])} messages")
    
    # Test LLM API connection
    logger.info("\n4️⃣ Testing LLM API Connection...")
    logger.info(f"   Attempting to connect to: {LLM_URL}")
    
    try:
        logger.info("   Sending test prompt to LLM...")
        test_prompt = "Say hello in one sentence."
        response = client._call_llm(test_prompt, max_tokens=20)
        logger.info(f"✅ LLM Response: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ LLM Connection Failed: {e}")
        logger.info("\n💡 To use LLM features, start LM Studio:")
        logger.info("   1. Download from https://lmstudio.ai")
        logger.info("   2. Load a model (e.g., Mistral, Llama 2)")
        logger.info("   3. Click 'Start Server'")
        logger.info(f"   4. Server should run on: {LLM_URL}")
        return False

if __name__ == "__main__":
    success = test_llm_client()
    logger.info("\n" + "=" * 60)
    logger.info(f"Result: {'✅ PASS' if success else '❌ LLM Not Available'}")
    logger.info("\nNote: LLM server is optional. The retriever and data processing")
    logger.info("      work without it. LLM is only needed for AI responses.")
