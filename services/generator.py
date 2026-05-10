import os
from groq import Groq
import config
import logging

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, query: str, contexts: list, chat_history: list = None, temperature: float = None):
        if temperature is None:
            temperature = config.TEMPERATURE
            
        # Assemble context from retrieved chunks
        context_text = "\n\n".join([f"Source ({i+1}): {doc['text']}" for i, doc in enumerate(contexts)])
        
        system_prompt = (
            "You are a helpful and intelligent industrial AI assistant.\n"
            "Use the provided context to answer the user's question accurately and comprehensively.\n"
            "If the context contains technical specs or data, present them clearly (use markdown tables if appropriate).\n"
            "If the context contains partial information, provide what you can and synthesize it logically.\n"
            "For general greetings, respond politely. If the question is completely unrelated to the context, politely state that you can only answer based on the documents.\n\n"
            "Context:\n"
            f"{context_text}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject conversation history for context memory (last 4 messages)
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
        # Append the current query
        messages.append({"role": "user", "content": query})
        
        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
