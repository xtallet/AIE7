#!/usr/bin/env python3
"""
Script to read Alice in Wonderland PDF and generate a random question
"""

import PyPDF2
import random

def read_alice_pdf():
    """Read the Alice in Wonderland PDF and return the text content"""
    try:
        with open('data/alice_in_wonderland.pdf', 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def generate_random_question():
    """Generate a random question about Alice in Wonderland"""
    questions = [
        "What was Alice doing when she first saw the White Rabbit?",
        "What did the White Rabbit say to itself when Alice first saw it?",
        "What did the White Rabbit take out of its waistcoat-pocket?",
        "What did Alice think about the book her sister was reading?",
        "What was Alice considering making when the White Rabbit appeared?",
        "What did Alice think was the use of a book?",
        "What did Alice think about the White Rabbit having a waistcoat-pocket?",
        "What did Alice think about the White Rabbit having a watch?",
        "What did Alice do when she saw the White Rabbit go down the rabbit-hole?",
        "What did Alice think about getting out of the rabbit-hole?",
        "What did Alice think about the hot day?",
        "What did Alice think about the White Rabbit's behavior?",
        "What did Alice think about the White Rabbit's appearance?",
        "What did Alice think about the White Rabbit's actions?",
        "What did Alice think about the White Rabbit's timing?"
    ]
    
    return random.choice(questions)

def main():
    """Main function to read PDF and generate question"""
    print("=== Alice in Wonderland Question Generator ===\n")
    
    # Read the PDF
    print("Reading Alice in Wonderland PDF...")
    content = read_alice_pdf()
    
    if content:
        print(f"✓ Successfully read PDF ({len(content)} characters)")
        print(f"✓ Content preview: {content[:200]}...")
        
        # Generate random question
        question = generate_random_question()
        print(f"\n🎯 Random Question: {question}")
        
        # Show how to use it with the RAG system
        print(f"\n💡 You can now test this question with your RAG system:")
        print(f"   vector_db.search_by_text('{question}', k=3)")
        
    else:
        print("✗ Failed to read PDF")

if __name__ == "__main__":
    main() 