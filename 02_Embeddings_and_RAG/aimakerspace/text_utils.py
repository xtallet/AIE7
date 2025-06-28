import os
from typing import List
import PyPDF2


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_file()
            elif self.path.endswith(".pdf"):
                self.load_pdf()
            else:
                raise ValueError(
                    "Provided file is not a supported format. Supported formats: .txt, .pdf"
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a supported file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf(self):
        """Load text content from a PDF file"""
        try:
            with open(self.path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
                
                # Clean up the text (remove excessive whitespace)
                text_content = " ".join(text_content.split())
                self.documents.append(text_content)
                
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):
                    # Load PDF files in directory
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text_content = ""
                            
                            for page_num in range(len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                text_content += page.extract_text() + "\n"
                            
                            text_content = " ".join(text_content.split())
                            self.documents.append(text_content)
                    except Exception as e:
                        print(f"Warning: Could not read PDF file {file_path}: {str(e)}")
                        continue

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
