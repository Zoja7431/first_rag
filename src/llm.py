"""
src/llm.py — GroqChat (llama3.1 + qwen + gpt-oss)
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from src.config import get_config

class LLMManager:
    _instance: Optional['LLMManager'] = None
    _llm = None
    _model_index = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self, model_index: int = 0):
        if self._llm is None or model_index != self._model_index:
            self._llm = self._init_llm(model_index)
            self._model_index = model_index
        return self._llm

    def _init_llm(self, model_index: int):
        config = get_config()
        models = config.llm.groq.models
        
        if model_index >= len(models):
            model_index = 0
            
        model_id = models[model_index]
        print(f"🚀 Groq LLM: {model_id}")
        
        return ChatGroq(
            groq_api_key=config.llm.groq.api_key,
            model_name=model_id,
            temperature=config.llm.groq.temperature,
            max_tokens=config.llm.groq.max_tokens,
        )

def get_llm(model_index: int = 0):
    return LLMManager().get_llm(model_index)

def get_llm_manager():
    return LLMManager()
