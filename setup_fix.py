"""
setup_fix.py - Automatic configuration fix

Run: python setup_fix.py (from first_rag/ folder)

This script:
1. Creates __init__.py files if missing
2. Fixes src/data_loader.py (removes overlap_size from RecursiveChunker)
3. Checks all necessary files are in place
4. Tests that imports work
"""

import os
from pathlib import Path


def create_init_files():
    """Create __init__.py files if missing"""
    init_files = [
        "src/__init__.py",
        "src/tests/__init__.py",
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if init_file == "src/__init__.py":
                path.write_text("""import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
""")
            else:
                path.write_text("")
            print(f"✅ Created: {init_file}")
        else:
            print(f"✅ Already exists: {init_file}")


def fix_data_loader():
    """Fix src/data_loader.py"""
    path = Path("src/data_loader.py")
    
    if not path.exists():
        print("❌ File src/data_loader.py not found!")
        return False
    
    content = path.read_text(encoding='utf-8')
    
    # Check if not already fixed
    if "overlap_size=tp.overlap_size" in content and "RecursiveChunker(" in content:
        # Find and replace
        old_code = """self.chunker = RecursiveChunker(
            chunk_size=tp.chunk_size,
            overlap_size=tp.overlap_size  # ✅ overlap_size (не chunk_overlap)
        )"""
        
        new_code = """self.chunker = RecursiveChunker(
            chunk_size=tp.chunk_size
        )
        self.overlap_size = tp.overlap_size"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            path.write_text(content, encoding='utf-8')
            print("✅ Fixed: src/data_loader.py (removed overlap_size from RecursiveChunker)")
            return True
        else:
            print("⚠️  src/data_loader.py has different structure, manual check needed")
            return False
    else:
        print("✅ src/data_loader.py already fixed (overlap_size removed)")
        return True


def check_chunk_analyse():
    """Check that chunk_analyse.py exists and has correct imports"""
    path = Path("src/tests/chunk_analyse.py")
    
    if path.exists():
        print("✅ Found: src/tests/chunk_analyse.py")
        
        # Check that it has correct imports
        content = path.read_text(encoding='utf-8')
        if "from src." in content or "import sys" in content:
            print("✅ File structure looks correct")
            return True
        else:
            print("⚠️  File exists but structure looks suspicious")
            return False
    else:
        print("⚠️  WARNING: src/tests/chunk_analyse.py not found!")
        print("   Please use the chunk_analyse.py from artifact")
        return False


def check_test_file():
    """Check that test_chunking_integration.py exists"""
    path = Path("src/tests/test_chunking_integration.py")
    
    if path.exists():
        print("✅ Found: src/tests/test_chunking_integration.py")
        return True
    else:
        print("⚠️  WARNING: src/tests/test_chunking_integration.py not found!")
        print("   Copy it from artifact or rename existing test file")
        return False


def test_imports():
    """Test that imports work"""
    print("\n📝 Testing imports...")
    try:
        from src.config import get_config
        print("✅ from src.config import get_config - OK")
    except ImportError as e:
        print(f"❌ from src.config import get_config - FAILED: {e}")
        return False
    
    try:
        from src.data_loader import get_pdf_chunker
        print("✅ from src.data_loader import get_pdf_chunker - OK")
    except ImportError as e:
        print(f"❌ from src.data_loader import get_pdf_chunker - FAILED: {e}")
        return False
    
    try:
        from src.embeddings import get_embeddings_manager
        print("✅ from src.embeddings import get_embeddings_manager - OK")
    except ImportError as e:
        print(f"❌ from src.embeddings import get_embeddings_manager - FAILED: {e}")
        return False
    
    return True


def main():
    print("="*80)
    print("🔧 AUTOMATIC CONFIGURATION FIX")
    print("="*80 + "\n")
    
    # Step 1
    print("📝 Step 1: Creating __init__.py files...")
    create_init_files()
    print()
    
    # Step 2
    print("📝 Step 2: Fixing src/data_loader.py...")
    loader_ok = fix_data_loader()
    print()
    
    # Step 3
    print("📝 Step 3: Checking chunk_analyse.py...")
    chunk_ok = check_chunk_analyse()
    print()
    
    # Step 4
    print("📝 Step 4: Checking test_chunking_integration.py...")
    test_ok = check_test_file()
    print()
    
    # Step 5
    import_ok = test_imports()
    print()
    
    print("="*80)
    
    if loader_ok and chunk_ok and test_ok and import_ok:
        print("✅ ALL FIXES COMPLETED SUCCESSFULLY!")
        print("\n📋 You can now run:")
        print("  1. pytest src/tests/test_chunking_integration.py -v -s")
        print("  2. python -m src.tests.chunk_analyse")
        print("\n✨ Imports work correctly from any location in the project!")
    else:
        print("⚠️  MANUAL INTERVENTION REQUIRED")
        if not loader_ok:
            print("\n  ❌ src/data_loader.py fix failed")
            print("     Check file manually - should remove overlap_size from RecursiveChunker")
        if not chunk_ok:
            print("\n  ❌ src/tests/chunk_analyse.py not found or has wrong structure")
            print("     Copy chunk_analyse.py from artifact to src/tests/")
        if not test_ok:
            print("\n  ❌ test_chunking_integration.py not found")
            print("     Copy it from artifact to src/tests/")
        if not import_ok:
            print("\n  ❌ Imports not working")
            print("     Check that __init__.py files were created correctly")
    
    print("="*80)
    
    return loader_ok and chunk_ok and test_ok and import_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)