"""
LLM client for WikiTalk
"""
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.url = LLM_URL
        self.model = LLM_MODEL
        self.temperature = TEMPERATURE
        logger.info(f"🤖 LLM Client initialized")
        logger.info(f"   URL: {self.url}")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Temperature: {self.temperature}")
        
    def query_rewrite(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
        """Rewrite query based on conversation history"""
        # OPTIMIZATION: Query rewriting disabled for speed
        # Queries are usually specific enough already, and this adds 0.5+ seconds per request
        # Set to True to re-enable
        QUERY_REWRITE_ENABLED = False
        
        if not QUERY_REWRITE_ENABLED:
            return query
            
        logger.debug(f"📝 Query rewrite requested for: '{query}'")
        
        if not conversation_history:
            logger.debug("   No conversation history, returning original query")
            return query
        
        # Get recent conversation context
        recent_history = conversation_history[-MEMORY_TURNS:]
        logger.debug(f"   Using last {len(recent_history)} messages from history")
        
        # Create context for query rewriting
        context_parts = []
        for turn in recent_history:
            if turn['role'] == 'user':
                context_parts.append(f"User: {turn['content']}")
            elif turn['role'] == 'assistant':
                context_parts.append(f"Assistant: {turn['content']}")
        
        context = "\n".join(context_parts)
        
        # Rewrite query to be more specific
        rewrite_prompt = f"""Based on the conversation history below, rewrite the user's latest query to be more specific and self-contained for Wikipedia search.

Conversation History:
{context}

User's latest query: {query}

Rewritten query (make it specific and searchable):"""

        try:
            logger.info(f"🔄 Attempting LLM query rewrite...")
            response = self._call_llm(rewrite_prompt, max_tokens=100)
            rewritten = response.strip()
            logger.info(f"   ✓ Query rewritten to: '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"⚠️ Query rewrite failed: {e}")
            logger.info(f"   Using original query instead")
            return query
    
    def generate_response(self, query: str, sources: List[Dict[str, Any]], 
                         conversation_history: List[Dict[str, str]]) -> str:
        """Generate response using retrieved sources"""
        logger.info(f"📚 Generating response for query: '{query}'")
        logger.info(f"   Using {len(sources)} sources")
        
        # Format sources for the prompt
        sources_text = self._format_sources_for_prompt(sources)
        
        # Create system prompt
        system_prompt = """You are a factual historian using provided Wikipedia sources. 
Answer the user's question based on the sources provided. 
Cite sources with [1], [2], etc. 
If information is missing from the sources, say so clearly.
Be conversational but accurate."""

        # Create user prompt
        user_prompt = f"""Question: {query}

Sources:
{sources_text}

Answer:"""

        # Add conversation context if available
        if conversation_history:
            recent_history = conversation_history[-MEMORY_TURNS:]
            logger.debug(f"   Including {len(recent_history)} recent history messages")
            context_parts = []
            for turn in recent_history:
                if turn['role'] == 'user':
                    context_parts.append(f"Previous question: {turn['content']}")
                elif turn['role'] == 'assistant':
                    context_parts.append(f"Previous answer: {turn['content']}")
            
            if context_parts:
                context = "\n".join(context_parts)
                user_prompt = f"""Context from previous conversation:
{context}

Current question: {query}

Sources:
{sources_text}

Answer:"""

        try:
            logger.info(f"🔄 Attempting LLM response generation...")
            response = self._call_llm(user_prompt, system_prompt=system_prompt)
            logger.info(f"   ✓ Response generated ({len(response)} chars)")
            return response.strip()
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            fallback = "I apologize, but I'm having trouble generating a response right now."
            return fallback
    
    def _format_sources_for_prompt(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for the LLM prompt"""
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            formatted_sources.append(f"[{i}] {source['title']}\n{source['text']}")
        return "\n\n".join(formatted_sources)
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, 
                  max_tokens: int = 1000) -> str:
        """Call the LLM API"""
        logger.debug(f"🌐 Preparing LLM API call")
        logger.debug(f"   Prompt length: {len(prompt)} chars")
        logger.debug(f"   Max tokens: {max_tokens}")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"   System prompt included ({len(system_prompt)} chars)")
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            logger.info(f"🚀 Sending request to {self.url}")
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            logger.debug(f"   Status code: {response.status_code}")
            
            result = response.json()
            logger.debug(f"   Response received")
            
            if 'choices' not in result or not result['choices']:
                logger.error(f"   No choices in response: {result}")
                raise Exception("No response choices in LLM response")
            
            content = result['choices'][0]['message']['content']
            logger.info(f"   ✓ LLM responded ({len(content)} chars)")
            return content
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Connection failed to {self.url}: {e}")
            logger.error(f"   Is LM Studio or llama.cpp running?")
            raise Exception(f"Cannot connect to LLM at {self.url}")
        except requests.exceptions.Timeout as e:
            logger.error(f"❌ LLM request timed out: {e}")
            raise Exception(f"LLM request timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LLM API call failed: {e}")
            raise Exception(f"LLM API call failed: {e}")
        except KeyError as e:
            logger.error(f"❌ Unexpected LLM response format: {e}")
            logger.error(f"   Response: {result}")
            raise Exception(f"Unexpected LLM response format: {e}")


class ConversationManager:
    def __init__(self):
        self.conversations_dir = CONVERSATIONS_DIR
        self.conversations_dir.mkdir(exist_ok=True)
        logger.info(f"💾 Conversation Manager initialized")
        logger.info(f"   Storage: {self.conversations_dir}")
    
    def save_conversation(self, session_id: str, history: List[Dict[str, str]], 
                         last_topic: str = None):
        """Save conversation to file"""
        logger.debug(f"💾 Saving conversation {session_id}")
        logger.debug(f"   History length: {len(history)} messages")
        
        conversation_data = {
            "session_id": session_id,
            "history": history,
            "last_topic": last_topic,
            "timestamp": datetime.now().isoformat()
        }
        
        file_path = self.conversations_dir / f"session_{session_id}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            logger.info(f"   ✓ Saved to {file_path}")
        except Exception as e:
            logger.error(f"   ✗ Failed to save: {e}")
    
    def load_conversation(self, session_id: str) -> Dict[str, Any]:
        """Load conversation from file"""
        logger.debug(f"📖 Loading conversation {session_id}")
        file_path = self.conversations_dir / f"session_{session_id}.json"
        
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"   ✓ Loaded {len(data.get('history', []))} messages")
                return data
            except Exception as e:
                logger.error(f"   ✗ Failed to load: {e}")
        else:
            logger.debug(f"   New session (no saved conversation)")
        
        return {
            "session_id": session_id,
            "history": [],
            "last_topic": None,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """Add a new exchange to the conversation"""
        logger.debug(f"➕ Adding exchange to {session_id}")
        
        conversation = self.load_conversation(session_id)
        
        conversation["history"].append({"role": "user", "content": user_message})
        conversation["history"].append({"role": "assistant", "content": assistant_message})
        logger.debug(f"   User: {user_message[:50]}...")
        logger.debug(f"   Assistant: {assistant_message[:50]}...")
        
        # Keep only recent history
        if len(conversation["history"]) > MEMORY_TURNS * 2:
            logger.debug(f"   Trimming history to {MEMORY_TURNS * 2} messages")
            conversation["history"] = conversation["history"][-MEMORY_TURNS * 2:]
        
        self.save_conversation(session_id, conversation["history"], conversation.get("last_topic"))


if __name__ == "__main__":
    logger.info("🧪 Running LLM client tests")
    
    # Test LLM client
    client = LLMClient()
    
    # Test query rewrite
    logger.info("\n📝 Testing query rewrite...")
    history = [
        {"role": "user", "content": "Tell me about World War I"},
        {"role": "assistant", "content": "World War I was a global war..."}
    ]
    
    rewritten = client.query_rewrite("What caused it?", history)
    logger.info(f"Original: 'What caused it?'")
    logger.info(f"Rewritten: '{rewritten}'")
    
    # Test conversation manager
    logger.info("\n💾 Testing conversation manager...")
    manager = ConversationManager()
    manager.add_exchange("test_session", "Hello", "Hi there!")
    conversation = manager.load_conversation("test_session")
    logger.info(f"Conversation has {len(conversation['history'])} messages")

