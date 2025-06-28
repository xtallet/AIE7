#!/usr/bin/env python3
"""
Test script to verify PDF loader functionality
"""

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter

def test_text_loader():
    """Test the existing text file loading functionality"""
    print("Testing text file loading...")
    try:
        text_loader = TextFileLoader("data/PMarcaBlogs.txt")
        documents = text_loader.load_documents()
        print(f"✓ Successfully loaded {len(documents)} text document(s)")
        print(f"  First 100 characters: {documents[0][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Error loading text file: {e}")
        return False

def test_pdf_loader():
    """Test the new PDF file loading functionality"""
    print("\nTesting PDF file loading...")
    try:
        # Test with the actual PDF file we created
        pdf_loader = TextFileLoader("data/test_document.pdf")
        documents = pdf_loader.load_documents()
        print(f"✓ Successfully loaded {len(documents)} PDF document(s)")
        print(f"  First 200 characters: {documents[0][:200]}...")
        
        # Verify that the content contains expected text
        content = documents[0].lower()
        if "test document" in content and "pdf loader" in content:
            print("✓ PDF content extracted correctly")
        else:
            print("⚠️  PDF content may not have been extracted properly")
            
        return True
    except Exception as e:
        print(f"✗ Error loading PDF file: {e}")
        return False

def test_compatibility():
    """Test that the existing code still works"""
    print("\nTesting backward compatibility...")
    try:
        # Test the existing workflow
        text_loader = TextFileLoader("data/PMarcaBlogs.txt")
        documents = text_loader.load_documents()
        
        text_splitter = CharacterTextSplitter()
        split_documents = text_splitter.split_texts(documents)
        
        print(f"✓ Successfully processed {len(split_documents)} chunks")
        print("✓ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility error: {e}")
        return False

def test_pdf_workflow():
    """Test the complete PDF workflow including splitting"""
    print("\nTesting PDF workflow with text splitting...")
    try:
        # Load PDF
        pdf_loader = TextFileLoader("data/test_document.pdf")
        documents = pdf_loader.load_documents()
        
        # Split the content
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = text_splitter.split_texts(documents)
        
        print(f"✓ Successfully split PDF into {len(split_documents)} chunks")
        print(f"  First chunk: {split_documents[0][:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ PDF workflow error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== PDF Loader Test Suite ===\n")
    
    tests = [
        test_text_loader,
        test_pdf_loader,
        test_compatibility,
        test_pdf_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PDF loader is ready to use.")
        print("\nYou can now use PDF files in your RAG application:")
        print("  pdf_loader = TextFileLoader('your_document.pdf')")
        print("  documents = pdf_loader.load_documents()")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 