import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def process_pdf_and_build_vectorstore(pdf_path: str, openai_api_key: str) -> object:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
    client = QdrantClient(':memory:')
    client.create_collection(
        collection_name='insurance_policy',
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name='insurance_policy',
        embedding=embeddings,
    )
    _ = vector_store.add_documents(documents=split_documents)
    return vector_store

from backend.agent_graph import run_agentic_rag 