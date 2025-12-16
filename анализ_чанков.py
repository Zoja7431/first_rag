import sys
from pathlib import Path
import csv
from typing import List, Dict, Tuple
import statistics

# sys.path fix
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.data_loader import get_pdf_chunker
from src.embeddings import get_embeddings_manager
from src.vector_store import get_vector_store_manager
import numpy as np
from langchain_core.documents import Document

# ============================================================================
# АНАЛИЗ ЧАНКОВ
# ============================================================================

class ChunkAnalyzer:
    """Детальный анализ качества чанков"""
    
    def __init__(self):
        self.config = get_config()
        self.chunker = get_pdf_chunker()
        self.chunks: List[Document] = []
        self.stats: Dict = {}
    
    def load_pdf_chunks(self) -> List[Document]:
        """Загружаем чанки из всех PDF"""
        pdf_dir = Path(self.config.data.pdf_path)
        pdf_files = list(pdf_dir.glob("*.pdf"))[:1]  # Первый PDF для анализа
        
        if not pdf_files:
            print("❌ Нет PDF файлов!")
            return []
        
        print(f"📚 Анализируем PDF: {pdf_files[0].name}")
        self.chunks = self.chunker.load_and_chunk_pdf(str(pdf_files[0]))
        print(f"✅ Загружено {len(self.chunks)} чанков\n")
        return self.chunks
    
    def analyze_chunk_sizes(self) -> Dict:
        """Анализ размеров чанков"""
        if not self.chunks:
            return {}
        
        char_counts = [len(c.page_content) for c in self.chunks]
        token_estimates = [len(c.page_content) // 4 for c in self.chunks]  # Грубая оценка
        
        stats = {
            'total_chunks': len(self.chunks),
            'total_chars': sum(char_counts),
            'total_tokens_est': sum(token_estimates),
            'char_mean': statistics.mean(char_counts),
            'char_median': statistics.median(char_counts),
            'char_stdev': statistics.stdev(char_counts) if len(char_counts) > 1 else 0,
            'char_min': min(char_counts),
            'char_max': max(char_counts),
            'token_mean_est': statistics.mean(token_estimates),
        }
        
        return stats
    
    def analyze_chunk_content_quality(self) -> Dict:
        """Анализ качества содержимого чанков"""
        if not self.chunks:
            return {}
        
        # Анализ первых 10 чанков
        sample_chunks = self.chunks[:10]
        
        quality_metrics = {
            'sample_size': len(sample_chunks),
            'chunks_with_content': 0,
            'chunks_under_50_chars': 0,
            'chunks_over_1000_chars': 0,
            'has_cyrillic': 0,
            'has_numbers': 0,
            'has_special_chars': 0,
        }
        
        for chunk in sample_chunks:
            text = chunk.page_content
            
            if text.strip():
                quality_metrics['chunks_with_content'] += 1
            
            if len(text) < 50:
                quality_metrics['chunks_under_50_chars'] += 1
            
            if len(text) > 1000:
                quality_metrics['chunks_over_1000_chars'] += 1
            
            if any('\u0400' <= char <= '\u04FF' for char in text):
                quality_metrics['has_cyrillic'] += 1
            
            if any(char.isdigit() for char in text):
                quality_metrics['has_numbers'] += 1
            
            if any(char in '.,;:!?-' for char in text):
                quality_metrics['has_special_chars'] += 1
        
        return quality_metrics
    
    def show_sample_chunks(self, n: int = 3):
        """Показываем примеры чанков"""
        print("\n" + "="*80)
        print("📄 ПРИМЕРЫ ЧАНКОВ (первые 3)")
        print("="*80)
        
        for i, chunk in enumerate(self.chunks[:n]):
            text = chunk.page_content
            meta = chunk.metadata
            
            print(f"\n{'─'*80}")
            print(f"Чанк #{meta.get('chunk_id', i)} из книги '{meta.get('book', '?')}' стр.{meta.get('page', '?')}")
            print(f"Размер: {len(text)} символов ({len(text)//4} токенов примерно)")
            print(f"{'─'*80}")
            print(f"{text[:300]}...")
            print(f"Metadata: {meta}")
    
    def save_statistics_to_csv(self, filename: str = "статистика_чанков.csv"):
        """Сохраняем статистику каждого чанка в CSV"""
        if not self.chunks:
            print("❌ Нет чанков для анализа!")
            return
        
        rows = []
        for chunk in self.chunks:
            text = chunk.page_content
            meta = chunk.metadata
            
            rows.append({
                'chunk_id': meta.get('chunk_id', ''),
                'book': meta.get('book', ''),
                'page': meta.get('page', ''),
                'chunk_index': meta.get('chunk_index', ''),
                'char_count': len(text),
                'token_count_est': len(text) // 4,
                'word_count': len(text.split()),
                'has_newline': '\n' in text,
                'preview': text[:100].replace('\n', ' '),
            })
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Статистика сохранена в {filename}")
    
    def print_statistics(self):
        """Вывод статистики в консоль"""
        size_stats = self.analyze_chunk_sizes()
        quality_stats = self.analyze_chunk_content_quality()
        
        print("\n" + "="*80)
        print("📊 СТАТИСТИКА ЧАНКОВ")
        print("="*80)
        
        print(f"\n📈 РАЗМЕРЫ ЧАНКОВ:")
        print(f"  Всего чанков: {size_stats.get('total_chunks', 0)}")
        print(f"  Всего символов: {size_stats.get('total_chars', 0):,}")
        print(f"  Примерно токенов: {size_stats.get('total_tokens_est', 0):,}")
        print(f"  Средний размер: {size_stats.get('char_mean', 0):.1f} символов")
        print(f"  Медианный размер: {size_stats.get('char_median', 0):.1f} символов")
        print(f"  Стд. отклонение: {size_stats.get('char_stdev', 0):.1f}")
        print(f"  Минимум: {size_stats.get('char_min', 0)} символов")
        print(f"  Максимум: {size_stats.get('char_max', 0)} символов")
        
        print(f"\n✅ КАЧЕСТВО СОДЕРЖИМОГО (первые 10 чанков):")
        print(f"  С содержимым: {quality_stats.get('chunks_with_content', 0)}/10")
        print(f"  <50 символов: {quality_stats.get('chunks_under_50_chars', 0)}/10 (плохо)")
        print(f"  >1000 символов: {quality_stats.get('chunks_over_1000_chars', 0)}/10 (может быть много)")
        print(f"  С кириллицей: {quality_stats.get('has_cyrillic', 0)}/10")
        print(f"  С цифрами: {quality_stats.get('has_numbers', 0)}/10")
        print(f"  С пунктуацией: {quality_stats.get('has_special_chars', 0)}/10")
        
        # Оценка качества
        avg_size = size_stats.get('char_mean', 0)
        if 200 <= avg_size <= 800:
            quality = "✅ ОТЛИЧНО"
        elif 100 <= avg_size <= 1500:
            quality = "🟡 ХОРОШО"
        else:
            quality = "❌ ПЛОХО"
        
        print(f"\n🎯 ОЦЕНКА КАЧЕСТВА: {quality}")
        print(f"  Оптимальный размер: 200-800 символов")
        print(f"  Текущий средний размер: {avg_size:.0f} символов")


# ============================================================================
# АНАЛИЗ ЭМБЕДДИНГОВ
# ============================================================================

class EmbeddingsAnalyzer:
    """Анализ качества эмбеддингов"""
    
    def __init__(self):
        self.config = get_config()
        try:
            self.embeddings_manager = get_embeddings_manager()
            self.embeddings_available = True
        except Exception as e:
            print(f"⚠️ Embeddings не доступны: {e}")
            self.embeddings_available = False
    
    def test_embeddings(self, texts: List[str] = None):
        """Тестируем эмбеддинги"""
        if not self.embeddings_available:
            print("⚠️ Embeddings менеджер не инициализирован")
            return
        
        if texts is None:
            texts = [
                "Психология изучает поведение человека",
                "Эмбеддинги преобразуют текст в векторы",
                "Qdrant хранит векторные данные",
            ]
        
        print("\n" + "="*80)
        print("🧠 АНАЛИЗ ЭМБЕДДИНГОВ")
        print("="*80)
        
        try:
            embeddings = self.embeddings_manager.embed_texts(texts)
            
            print(f"\n📊 Модель: {self.config.embeddings.model_name}")
            print(f"📊 Устройство: {self.config.embeddings.device}")
            print(f"📊 Размерность: {self.config.embeddings.embedding_dim}")
            
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                emb_array = np.array(emb)
                print(f"\n✅ Текст {i+1}: '{text[:60]}...'")
                print(f"   Размер вектора: {len(emb_array)}")
                print(f"   Норма (L2): {np.linalg.norm(emb_array):.4f}")
                print(f"   Первые 5 элементов: {emb_array[:5]}")
            
            # Сравнение похожести
            if len(embeddings) >= 2:
                emb_arrays = [np.array(e) for e in embeddings]
                similarity = np.dot(emb_arrays[0], emb_arrays[1])
                print(f"\n🔗 Косинусное сходство между текстом 1 и 2: {similarity:.4f}")
                print(f"   (1.0 = идентичны, 0.0 = разные, -1.0 = противоположны)")
        
        except Exception as e:
            print(f"❌ Ошибка при создании эмбеддингов: {e}")


# ============================================================================
# АНАЛИЗ VECTOR STORE
# ============================================================================

class VectorStoreAnalyzer:
    """Анализ векторной базы данных"""
    
    def __init__(self):
        self.config = get_config()
        try:
            self.vs_manager = get_vector_store_manager()
            self.available = True
        except Exception as e:
            print(f"⚠️ Vector Store не доступен: {e}")
            self.available = False
    
    def check_collection_status(self):
        """Проверяем статус коллекции в Qdrant"""
        if not self.available:
            print("⚠️ Vector Store менеджер не инициализирован")
            return
        
        print("\n" + "="*80)
        print("📦 СТАТУС VECTOR STORE (QDRANT)")
        print("="*80)
        
        try:
            stats = self.vs_manager.get_collection_stats()
            
            print(f"\n✅ Коллекция: {stats.get('collection_name', '?')}")
            print(f"   URL: {self.config.qdrant.full_url}")
            print(f"   Документов (points): {stats.get('points_count', 0)}")
            print(f"   Векторов: {stats.get('vectors_count', 0)}")
            print(f"   Сегментов: {stats.get('segments_count', 0)}")
            
            if stats.get('points_count', 0) > 0:
                print(f"\n🎉 Документы УЖЕ ЗАГРУЖЕНЫ в vector store!")
            else:
                print(f"\n⚠️ Vector store ПУСТ. Документы еще не загружены.")
                print(f"   Используй: vector_store_manager.add_documents(chunks)")
        
        except Exception as e:
            print(f"❌ Ошибка при проверке статуса: {e}")
    
    def show_add_chunks_example(self):
        """Показываем как добавить чанки в vector store"""
        print("\n" + "="*80)
        print("💾 КАК ДОБАВИТЬ ЧАНКИ В VECTOR STORE")
        print("="*80)
        
        example = """
from src.data_loader import get_pdf_chunker
from src.vector_store import get_vector_store_manager

# 1. Загружаем чанки
chunker = get_pdf_chunker()
chunks = chunker.load_multiple(['path/to/pdf1.pdf', 'path/to/pdf2.pdf'])

# 2. Добавляем в vector store
vs_manager = get_vector_store_manager()
ids = vs_manager.add_documents(chunks)

print(f"✅ Добавлено {len(ids)} документов!")

# 3. Проверяем статус
stats = vs_manager.get_collection_stats()
print(f"Всего документов в vector store: {stats['points_count']}")

# 4. Поиск по запросу
results = vs_manager.search("психология поведения", k=5)
for doc in results:
    print(f"✅ {doc.page_content[:100]}...")
"""
        print(example)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🔍 ПОЛНЫЙ АНАЛИЗ СИСТЕМЫ ЧАНКИРОВАНИЯ И ЭМБЕДДИНГОВ\n")
    
    # 1. Анализ чанков
    chunk_analyzer = ChunkAnalyzer()
    chunk_analyzer.load_pdf_chunks()
    chunk_analyzer.print_statistics()
    chunk_analyzer.show_sample_chunks(n=3)
    chunk_analyzer.save_statistics_to_csv()
    
    # 2. Анализ эмбеддингов
    emb_analyzer = EmbeddingsAnalyzer()
    emb_analyzer.test_embeddings()
    
    # 3. Статус vector store
    vs_analyzer = VectorStoreAnalyzer()
    vs_analyzer.check_collection_status()
    vs_analyzer.show_add_chunks_example()
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("="*80)


if __name__ == "__main__":
    main()