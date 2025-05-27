# PDF ChatBot

A PDF document question-answering and summarization assistant powered by large language models and semantic search. Upload a PDF, the app extracts and chunks text, creates embeddings for retrieval, and lets you chat with or summarize the document interactively.

---

## Features

- Extracts and cleans text from PDFs
- Splits text into sentence-aware chunks with overlap
- Builds a vector store using sentence embeddings and FAISS
- Supports retrieval-augmented generation (RAG) for accurate answers
- Provides summarization of the document or specific chunks
- Interactive chat interface using Gradio

---

## Pipeline Overview

1. **PDF Upload & Text Extraction**  
   The uploaded PDF file is processed with PyMuPDF (`fitz`) to extract the raw text.

2. **Text Chunking**  
   The extracted text is split into manageable, sentence-aware chunks with some overlap for context preservation.

3. **Embedding Generation & Vector Store**  
   Sentence embeddings are generated using `sentence-transformers`, and FAISS builds a vector index for similarity search.

4. **Query & Response Generation**  
   User queries are embedded and matched against the vector store to retrieve relevant chunks. The retrieved context is then used to generate accurate answers or summaries via a large language model (`Open-Orca/Mistral-7B-OpenOrca`).

5. **Interactive Chat Interface**  
   Users can ask questions or request summaries through a simple Gradio-powered web UI.

---

## Installation

1. Clone the repository:
  ```bash
   git clone https://github.com/M-e-h-18/offline-pdf-chatbot.git
   cd offline-pdf-chatbot
```

2. Create and activate a Python environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

Run the app locally with:
```bash
python main.py
```
This will start a Gradio web interface. Open the provided local URL in your browser, upload a PDF, and start interacting!

## Requirements

    Python 3.8+

    PyTorch with CUDA (optional but recommended for faster inference)

    Dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

    Hugging Face Transformers

    Sentence Transformers

    FAISS

    Gradio

    PyMuPDF (fitz)

