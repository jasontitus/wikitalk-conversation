"""
Main WikiTalk application
"""
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Disable tokenizers parallelism warnings (happens when TTS forks after model load)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from retriever import HybridRetriever
from llm_client import LLMClient, ConversationManager
from tts_client import TTSClient
from config import *


class WikiTalk:
    def __init__(self, use_embeddings=True):
        """Initialize WikiTalk
        
        Args:
            use_embeddings: If True, use semantic search; if False, use keyword search
        """
        # Initialize with embedding search by default (if available)
        self.retriever = HybridRetriever(use_bm25_only=not use_embeddings)
        self.llm_client = LLMClient()
        self.tts_client = TTSClient()
        self.conversation_manager = ConversationManager()
        self.session_id = str(uuid.uuid4())
        self.use_embeddings = use_embeddings
        
    def initialize(self):
        """Initialize all components"""
        print("Initializing WikiTalk...")
        
        try:
            self.retriever.load_indexes()
            if self.use_embeddings and self.retriever.faiss_index is not None:
                print("✓ Retrieval system loaded (using semantic search)")
            else:
                print("✓ Retrieval system loaded (using keyword search)")
        except Exception as e:
            print(f"✗ Failed to load retrieval system: {e}")
            return False
        
        try:
            tts_available = self.tts_client.test_tts()
            if tts_available:
                print("✓ TTS system ready")
            else:
                print("⚠ TTS system not available, will use text-only mode")
        except Exception as e:
            print(f"⚠ TTS system error: {e}")
        
        print("WikiTalk initialized successfully!")
        return True
    
    def process_query(self, query: str, use_tts: bool = True) -> Dict[str, Any]:
        
        start_time = time.time()
        
        # Load conversation history
        conversation = self.conversation_manager.load_conversation(self.session_id)
        history = conversation.get("history", [])
        
        # Rewrite query based on conversation context
        rewritten_query = self.llm_client.query_rewrite(query, history)
        print(f"Original query: {query}")
        print(f"Rewritten query: {rewritten_query}")
        
        # Retrieve relevant sources using appropriate search method
        print("Searching Wikipedia...")
        search_method = "embedding" if (self.use_embeddings and self.retriever.faiss_index) else "like"
        sources = self.retriever.search(rewritten_query, top_k=5, method=search_method)
        
        if not sources:
            response = "I couldn't find relevant information about that topic in my Wikipedia database."
        else:
            # Generate response using LLM
            print("Generating response...")
            response = self.llm_client.generate_response(
                rewritten_query, sources, history
            )
        
        # Add exchange to conversation
        self.conversation_manager.add_exchange(
            self.session_id, query, response
        )
        
        # Speak response if TTS is enabled
        if use_tts and response:
            print("Speaking response...")
            self.tts_client.speak(response)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "response": response,
            "sources": sources[:3],  # Top 3 sources
            "processing_time": processing_time,
            "session_id": self.session_id,
            "search_method": search_method
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        conversation = self.conversation_manager.load_conversation(self.session_id)
        return conversation.get("history", [])
    
    def clear_conversation(self):
        """Clear current conversation"""
        self.session_id = str(uuid.uuid4())
        print(f"New conversation started: {self.session_id}")
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        print("\n" + "="*60)
        print("🤖 WikiTalk - Local Conversational Historian")
        print("="*60)
        print("Ask me anything about history, science, or culture!")
        print("Type 'quit' to exit, 'clear' to start new conversation")
        print("="*60)
        
        while True:
            try:
                query = input("\n👤 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() in ['clear', 'new']:
                    self.clear_conversation()
                    continue
                
                if not query:
                    continue
                
                print("\n🤖 WikiTalk: ", end="", flush=True)
                
                # Process query
                result = self.process_query(query, use_tts=True)
                
                # Display response
                print(result["response"])
                
                # Show sources
                if result["sources"]:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. {source['title']}")
                
                print(f"\n⏱️  Processed in {result['processing_time']:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def close(self):
        """Clean up resources"""
        self.retriever.close()


def main():
    """Main entry point"""
    wikitalk = WikiTalk()
    
    if not wikitalk.initialize():
        print("Failed to initialize WikiTalk. Please check your setup.")
        return
    
    try:
        wikitalk.interactive_mode()
    finally:
        wikitalk.close()


if __name__ == "__main__":
    main()

