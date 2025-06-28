#!/usr/bin/env python3
"""
Test script to verify PDF loader with Alice in Wonderland PDF
"""

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter

def test_alice_pdf():
    """Test loading the Alice in Wonderland PDF"""
    print("=== Testing PDF Loader with Alice in Wonderland ===\n")
    
    try:
        # Load the PDF file
        pdf_loader = TextFileLoader("data/alice_in_wonderland.pdf")
        documents = pdf_loader.load_documents()
        
        print(f"✓ Successfully loaded {len(documents)} PDF document(s)")
        print(f"✓ Document length: {len(documents[0])} characters")
        
        # Show first 200 characters
        print(f"\nFirst 200 characters:")
        print("-" * 50)
        print(documents[0][:200])
        print("-" * 50)
        
        # Test text splitting
        print(f"\n=== Testing Text Splitting ===")
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_documents = text_splitter.split_texts(documents)
        
        print(f"✓ Successfully split into {len(split_documents)} chunks")
        print(f"✓ First chunk length: {len(split_documents[0])} characters")
        
        # Show first chunk
        print(f"\nFirst chunk:")
        print("-" * 50)
        print(split_documents[0])
        print("-" * 50)
        
        # Test that it contains Alice content
        if "Alice" in documents[0] and "Wonderland" in documents[0]:
            print("✓ PDF content verification: Contains 'Alice' and 'Wonderland'")
        else:
            print("⚠️  PDF content verification: Missing expected keywords")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing PDF loader: {e}")
        return False

def test_backward_compatibility():
    """Test that existing text file functionality still works"""
    print(f"\n=== Testing Backward Compatibility ===")
    
    try:
        # Test with existing text file
        text_loader = TextFileLoader("data/PMarcaBlogs.txt")
        documents = text_loader.load_documents()
        
        print(f"✓ Successfully loaded {len(documents)} text document(s)")
        print(f"✓ Text file functionality maintained")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing backward compatibility: {e}")
        return False

def main():
    """Run all tests"""
    tests = [
        test_alice_pdf,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PDF loader is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 