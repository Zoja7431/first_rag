"""
test_groq_models.py — Сравнение 5 Groq моделей
"""

from src.llm import get_llm
from src.rag_graph import ask
import time

QUESTION = "Как читать язык телодвижений по Пизу?"

models = [0,1,2,3,4]  # Индексы из config

print("🤖 GROQ RAG ТЕСТ (5 моделей)\n" + "="*60)

for i, model_idx in enumerate(models):
    print(f"\n🧪 МОДЕЛЬ {i+1}: {get_llm(model_idx).model_name}")
    
    start = time.time()
    answer = ask(QUESTION)
    elapsed = time.time() - start
    
    print(f"⏱️  {elapsed:.1f}s")
    print(f"📝 {answer[:200]}...")
    print("-"*60)

print("\n✅ Готово! Выбери лучшую модель для config!")
