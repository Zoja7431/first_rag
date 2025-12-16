"""
chunk_analyse.py - Deep analysis of chunking quality and embeddings

Run: python -m src.tests.chunk_analyse (from first_rag/ folder)
     or: python src/tests/chunk_analyse.py
"""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import csv
from typing import List, Dict
import statistics

from src.config import get_config
from src.data_loader import get_pdf_chunker
from src.embeddings import get_embeddings_manager
from src.vector_store import get_vector_store_manager
import numpy as np
from langchain_core.documents import Document


# ============================================================================
# CHUNK ANALYSIS
# ============================================================================

class ChunkAnalyzer:
    """Detailed analysis of chunk quality"""
    
    def __init__(self):
        self.config = get_config()
        self.chunker = get_pdf_chunker()
        self.chunks: List[Document] = []
        self.stats: Dict = {}
    
    def load_pdf_chunks(self) -> List[Document]:
        """Load chunks from all PDFs"""
        pdf_dir = Path(self.config.data.pdf_path)
        pdf_files = list(pdf_dir.glob("*.pdf"))[:1]  # First PDF for analysis
        
        if not pdf_files:
            print("❌ No PDF files found!")
            return []
        
        print(f"📚 Analyzing PDF: {pdf_files[0].name}")
        self.chunks = self.chunker.load_and_chunk_pdf(str(pdf_files[0]))
        print(f"✅ Loaded {len(self.chunks)} chunks\n")
        return self.chunks
    
    def analyze_chunk_sizes(self) -> Dict:
        """Analyze chunk sizes"""
        if not self.chunks:
            return {}
        
        char_counts = [len(c.page_content) for c in self.chunks]
        token_estimates = [len(c.page_content) // 4 for c in self.chunks]  # Rough estimate
        
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
        """Analyze content quality"""
        if not self.chunks:
            return {}
        
        # Analyze first 10 chunks
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
        """Show example chunks"""
        print("\n" + "="*80)
        print("📄 EXAMPLE CHUNKS (first 3)")
        print("="*80)
        
        for i, chunk in enumerate(self.chunks[:n]):
            text = chunk.page_content
            meta = chunk.metadata
            
            print(f"\n{'─'*80}")
            print(f"Chunk #{meta.get('chunk_id', i)} from '{meta.get('book', '?')}' page {meta.get('page', '?')}")
            print(f"Size: {len(text)} chars ({len(text)//4} tokens approx)")
            print(f"{'─'*80}")
            print(f"{text[:300]}...")
            print(f"Metadata: {meta}")
    
    def save_statistics_to_csv(self, filename: str = "chunk_statistics.csv"):
        """Save statistics for each chunk to CSV"""
        if not self.chunks:
            print("❌ No chunks to analyze!")
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
        
        print(f"✅ Statistics saved to {filename}")
    
    def print_statistics(self):
        """Print statistics to console"""
        size_stats = self.analyze_chunk_sizes()
        quality_stats = self.analyze_chunk_content_quality()
        
        print("\n" + "="*80)
        print("📊 CHUNK STATISTICS")
        print("="*80)
        
        print(f"\n📈 CHUNK SIZES:")
        print(f"  Total chunks: {size_stats.get('total_chunks', 0)}")
        print(f"  Total chars: {size_stats.get('total_chars', 0):,}")
        print(f"  Approx tokens: {size_stats.get('total_tokens_est', 0):,}")
        print(f"  Average size: {size_stats.get('char_mean', 0):.1f} chars")
        print(f"  Median size: {size_stats.get('char_median', 0):.1f} chars")
        print(f"  Std deviation: {size_stats.get('char_stdev', 0):.1f}")
        print(f"  Min: {size_stats.get('char_min', 0)} chars")
        print(f"  Max: {size_stats.get('char_max', 0)} chars")
        
        print(f"\n✅ CONTENT QUALITY (first 10 chunks):")
        print(f"  With content: {quality_stats.get('chunks_with_content', 0)}/10")
        print(f"  <50 chars: {quality_stats.get('chunks_under_50_chars', 0)}/10 (bad)")
        print(f"  >1000 chars: {quality_stats.get('chunks_over_1000_chars', 0)}/10 (may be too much)")
        print(f"  With Cyrillic: {quality_stats.get('has_cyrillic', 0)}/10")
        print(f"  With numbers: {quality_stats.get('has_numbers', 0)}/10")
        print(f"  With punctuation: {quality_stats.get('has_special_chars', 0)}/10")
        
        # Quality assessment
        avg_size = size_stats.get('char_mean', 0)
        if 200 <= avg_size <= 800:
            quality = "✅ EXCELLENT"
        elif 100 <= avg_size <= 1500:
            quality = "🟡 GOOD"
        else:
            quality = "❌ POOR"
        
        print(f"\n🎯 QUALITY ASSESSMENT: {quality}")
        print(f"  Optimal size: 200-800 chars")
        print(f"  Current average: {avg_size:.0f} chars")


# ============================================================================
# EMBEDDINGS ANALYSIS
# ============================================================================

class EmbeddingsAnalyzer:
    """Analyze embeddings quality"""
    
    def __init__(self):
        self.config = get_config()
        try:
            self.embeddings_manager = get_embeddings_manager()
            self.embeddings_available = True
        except Exception as e:
            print(f"⚠️ Embeddings not available: {e}")
            self.embeddings_available = False
    
    def test_embeddings(self, texts: List[str] = None):
        """Test embeddings"""
        if not self.embeddings_available:
            print("⚠️ Embeddings manager not initialized")
            return
        
        if texts is None:
            texts = [
                "Psychology studies human behavior",
                "Embeddings transform text into vectors",
                "Qdrant stores vector data",
            ]
        
        print("\n" + "="*80)
        print("🧠 EMBEDDINGS ANALYSIS")
        print("="*80)
        
        try:
            embeddings = self.embeddings_manager.embed_texts(texts)
            
            print(f"\n📊 Model: {self.config.embeddings.model_name}")
            print(f"📊 Device: {self.config.embeddings.device}")
            print(f"📊 Dimensions: {self.config.embeddings.embedding_dim}")
            
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                emb_array = np.array(emb)
                print(f"\n✅ Text {i+1}: '{text[:60]}...'")
                print(f"   Vector size: {len(emb_array)}")
                print(f"   L2 norm: {np.linalg.norm(emb_array):.4f}")
                print(f"   First 5 elements: {emb_array[:5]}")
            
            # Compare similarity
            if len(embeddings) >= 2:
                emb_arrays = [np.array(e) for e in embeddings]
                similarity = np.dot(emb_arrays[0], emb_arrays[1])
                print(f"\n🔗 Cosine similarity between text 1 and 2: {similarity:.4f}")
                print(f"   (1.0 = identical, 0.0 = different, -1.0 = opposite)")
        
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")


# ============================================================================
# VECTOR STORE ANALYSIS
# ============================================================================

class VectorStoreAnalyzer:
    """Analyze vector database"""
    
    def __init__(self):
        self.config = get_config()
        try:
            self.vs_manager = get_vector_store_manager()
            self.available = True
        except Exception as e:
            print(f"⚠️ Vector Store not available: {e}")
            self.available = False
    
    def check_collection_status(self):
        """Check Qdrant collection status"""
        if not self.available:
            print("⚠️ Vector Store manager not initialized")
            return
        
        print("\n" + "="*80)
        print("📦 VECTOR STORE STATUS (QDRANT)")
        print("="*80)
        
        try:
            stats = self.vs_manager.get_collection_stats()
            
            print(f"\n✅ Collection: {stats.get('collection_name', '?')}")
            print(f"   URL: {self.config.qdrant.full_url}")
            print(f"   Documents (points): {stats.get('points_count', 0)}")
            print(f"   Vectors: {stats.get('vectors_count', 0)}")
            print(f"   Segments: {stats.get('segments_count', 0)}")
            
            if stats.get('points_count', 0) > 0:
                print(f"\n🎉 Documents ALREADY LOADED in vector store!")
            else:
                print(f"\n⚠️ Vector store is EMPTY. Documents not loaded yet.")
                print(f"   Use: vector_store_manager.add_documents(chunks)")
        
        except Exception as e:
            print(f"❌ Error checking status: {e}")
    
    def show_add_chunks_example(self):
        """Show how to add chunks to vector store"""
        print("\n" + "="*80)
        print("💾 HOW TO ADD CHUNKS TO VECTOR STORE")
        print("="*80)
        
        example = """
from src.data_loader import get_pdf_chunker
from src.vector_store import get_vector_store_manager

# 1. Load chunks
chunker = get_pdf_chunker()
chunks = chunker.load_and_chunk_pdf('path/to/pdf.pdf')

# 2. Add to vector store
vs_manager = get_vector_store_manager()
ids = vs_manager.add_documents(chunks)

print(f"✅ Added {len(ids)} documents!")

# 3. Check status
stats = vs_manager.get_collection_stats()
print(f"Total documents in vector store: {stats['points_count']}")

# 4. Search by query
results = vs_manager.search("psychology behavior", k=5)
for doc in results:
    print(f"✅ {doc.page_content[:100]}...")
"""
        print(example)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🔍 FULL SYSTEM ANALYSIS - CHUNKING AND EMBEDDINGS\n")
    
    # 1. Analyze chunks
    chunk_analyzer = ChunkAnalyzer()
    chunk_analyzer.load_pdf_chunks()
    chunk_analyzer.print_statistics()
    chunk_analyzer.show_sample_chunks(n=3)
    chunk_analyzer.save_statistics_to_csv()
    
    # 2. Analyze embeddings
    emb_analyzer = EmbeddingsAnalyzer()
    emb_analyzer.test_embeddings()
    
    # 3. Vector store status
    vs_analyzer = VectorStoreAnalyzer()
    vs_analyzer.check_collection_status()
    vs_analyzer.show_add_chunks_example()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()