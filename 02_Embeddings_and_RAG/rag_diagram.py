#!/usr/bin/env python3
"""
Simple RAG Process Diagram Generator
Creates a visual representation of the RAG (Retrieval Augmented Generation) process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_rag_diagram():
    """Create a simple diagram of the RAG process"""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'processing': '#F3E5F5',  # Light purple
        'storage': '#E8F5E8',     # Light green
        'output': '#FFF3E0',      # Light orange
        'arrow': '#424242'        # Dark gray
    }
    
    # Title
    ax.text(5, 9.5, 'RAG (Retrieval Augmented Generation) Process', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Step 1: Document Loading
    doc_box = FancyBboxPatch((0.5, 7.5), 2, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['input'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(doc_box)
    ax.text(1.5, 8, '1. Document\nLoading', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 2: Text Splitting
    split_box = FancyBboxPatch((3.5, 7.5), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['processing'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(split_box)
    ax.text(4.5, 8, '2. Text\nSplitting', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 3: Embedding Generation
    embed_box = FancyBboxPatch((6.5, 7.5), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['processing'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(embed_box)
    ax.text(7.5, 8, '3. Embedding\nGeneration', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 4: Vector Database Storage
    db_box = FancyBboxPatch((2, 5.5), 6, 1, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['storage'], 
                           edgecolor='black', linewidth=2)
    ax.add_patch(db_box)
    ax.text(5, 6, '4. Vector Database Storage\n(Text + Embeddings + Metadata)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 5: User Query
    query_box = FancyBboxPatch((0.5, 4), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(query_box)
    ax.text(1.5, 4.5, '5. User\nQuery', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 6: Query Embedding
    query_embed_box = FancyBboxPatch((3.5, 4), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['processing'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(query_embed_box)
    ax.text(4.5, 4.5, '6. Query\nEmbedding', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 7: Similarity Search
    search_box = FancyBboxPatch((6.5, 4), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['processing'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(search_box)
    ax.text(7.5, 4.5, '7. Similarity\nSearch', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 8: Context Retrieval
    context_box = FancyBboxPatch((2, 2.5), 6, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['processing'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(context_box)
    ax.text(5, 3, '8. Context Retrieval\n(Relevant Text Chunks)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 9: LLM Generation
    llm_box = FancyBboxPatch((2, 0.5), 6, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['output'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(llm_box)
    ax.text(5, 1, '9. LLM Generation\n(Context + Query → Answer)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows
    arrows = [
        # Document loading flow
        ((2.5, 8), (3.5, 8)),      # 1 → 2
        ((5.5, 8), (6.5, 8)),      # 2 → 3
        ((7.5, 7.5), (5, 6.5)),    # 3 → 4
        
        # Query flow
        ((1.5, 4), (3.5, 4)),      # 5 → 6
        ((5.5, 4), (6.5, 4)),      # 6 → 7
        ((7.5, 4), (5, 3.5)),      # 7 → 8
        ((5, 2.5), (5, 1.5)),      # 8 → 9
        
        # Database connection
        ((5, 5.5), (5, 4.5)),      # 4 → 7 (database lookup)
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
        ax.add_patch(arrow)
    
    # Add some explanatory text
    ax.text(0.5, 9, 'Document Processing Phase', fontsize=14, fontweight='bold', color='blue')
    ax.text(0.5, 3.8, 'Query Processing Phase', fontsize=14, fontweight='bold', color='red')
    
    # Add metadata note
    ax.text(8.5, 6.2, 'Metadata\nSupport', fontsize=10, style='italic', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_simple_text_diagram():
    """Create a simple text-based diagram"""
    
    diagram = """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    RAG (Retrieval Augmented Generation) Process              │
    └─────────────────────────────────────────────────────────────────────────────┘

    📄 DOCUMENT PROCESSING PHASE
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
    │  1. Load    │───▶│  2. Split   │───▶│  3. Embed   │───▶│  4. Store in Vector │
    │ Documents   │    │   Text      │    │   Text      │    │   Database          │
    │ (PDF/TXT)   │    │   Chunks    │    │   Vectors   │    │   + Metadata        │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘
                                                                    │
                                                                    ▼
    🔍 QUERY PROCESSING PHASE
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
    │  5. User    │───▶│  6. Embed   │───▶│  7. Search  │───▶│  8. Retrieve        │
    │   Query     │    │   Query     │    │  Similar    │    │   Relevant          │
    │             │    │   Vector    │    │  Vectors    │    │   Context           │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘
                                                                    │
                                                                    ▼
    🤖 GENERATION PHASE
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  9. LLM Generation                                                          │
    │  Context + Query → Answer                                                   │
    │  (Using Chain-of-Thought prompting for detailed responses)                 │
    └─────────────────────────────────────────────────────────────────────────────┘

    🔧 ENHANCEMENTS IMPLEMENTED:
    • PDF Support (PyPDF2 integration)
    • Multiple Distance Metrics (Cosine + Euclidean)
    • Metadata Support (chunk_id, source_file, chunk_size, etc.)
    • Enhanced Prompting Strategies (Chain-of-Thought)
    """
    
    return diagram

def save_diagram():
    """Create and save the RAG diagram"""
    
    # Create the visual diagram
    fig = create_rag_diagram()
    
    # Save as PNG
    fig.savefig('rag_process_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Visual diagram saved as 'rag_process_diagram.png'")
    
    # Create and save text diagram
    text_diagram = create_simple_text_diagram()
    with open('rag_process_diagram.txt', 'w') as f:
        f.write(text_diagram)
    print("✅ Text diagram saved as 'rag_process_diagram.txt'")
    
    # Display the diagram
    plt.show()
    
    return text_diagram

if __name__ == "__main__":
    print("Creating RAG Process Diagram...")
    diagram_text = save_diagram()
    print("\n" + "="*80)
    print("TEXT DIAGRAM:")
    print("="*80)
    print(diagram_text) 