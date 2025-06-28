#!/usr/bin/env python3
"""
Script to create a test PDF file for testing the PDF loader
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_test_pdf():
    """Create a test PDF file with sample content"""
    
    # Create the PDF document
    doc = SimpleDocTemplate("data/test_document.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create the story (content) for the PDF
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    title = Paragraph("Test Document for PDF Loader", title_style)
    story.append(title)
    
    # Sample content
    content = """
    This is a test document created to verify the PDF loading functionality in our RAG application.
    
    The document contains multiple paragraphs to test how well the PDF loader extracts text content
    from different parts of the document. This is important for maintaining compatibility with the
    existing text processing pipeline.
    
    Key features to test:
    - Text extraction from multiple pages
    - Handling of different text formatting
    - Compatibility with the existing TextFileLoader interface
    - Integration with the CharacterTextSplitter
    
    The goal is to ensure that PDF files can be processed in the same way as text files,
    maintaining the same output format for the vector database and RAG pipeline.
    """
    
    # Split content into paragraphs
    paragraphs = content.strip().split('\n\n')
    
    for para in paragraphs:
        if para.strip():
            p = Paragraph(para.strip(), styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 12))
    
    # Add some additional content to test multi-page handling
    additional_content = """
    Additional content for testing purposes. This paragraph contains information about
    the RAG (Retrieval Augmented Generation) system and how it processes different types
    of documents including text files and now PDF files.
    
    The system should be able to:
    1. Load PDF documents
    2. Extract text content
    3. Split the content into chunks
    4. Generate embeddings
    5. Store in the vector database
    6. Enable semantic search and retrieval
    
    This ensures that users can work with both text and PDF documents seamlessly
    in their RAG applications.
    """
    
    additional_paragraphs = additional_content.strip().split('\n\n')
    for para in additional_paragraphs:
        if para.strip():
            p = Paragraph(para.strip(), styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(story)
    print("✓ Test PDF created: data/test_document.pdf")

if __name__ == "__main__":
    create_test_pdf() 