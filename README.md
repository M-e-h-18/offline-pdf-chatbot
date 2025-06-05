# PDF Analysis Suite

A comprehensive PDF document analysis tool that combines advanced AI capabilities for document comparison, question-answering, and summarization. This application leverages large language models, semantic search, and OCR technology to provide deep insights into PDF documents.

## ✨ Features

### 🔍 PDF Comparison
- **Side-by-side document comparison** with detailed similarity analysis
- **Visual diff highlighting** showing exact changes between documents
- **AI-powered analysis** identifying key differences and similarities
- **Comprehensive metrics** including word, character, and line similarity scores

### 💬 Interactive PDF Chat
- **Question-answering** with context-aware responses using RAG (Retrieval-Augmented Generation)
- **Semantic search** to find the most relevant content for your queries
- **Chat history** for continuous conversation with your documents
- **Smart text chunking** with sentence-aware segmentation

### 📝 Document Summarization
- **Adaptive summarization** strategy based on document length
- **Section-wise processing** for long documents
- **Comprehensive summaries** with key points and structured output
- **Progress tracking** with cancellation support

### 🔧 Advanced Processing
- **OCR fallback** for image-based PDF pages using Tesseract
- **Model quantization** for efficient memory usage (4-bit quantization)
- **GPU acceleration** with CUDA support
- **Robust error handling** with graceful degradation

## 🏗️ Architecture Overview

### 1. **Document Processing Pipeline**
- **Text Extraction**: PyMuPDF (`fitz`) with OCR fallback using `pytesseract`
- **Smart Chunking**: Context-preserving text segmentation with configurable overlap
- **Preprocessing**: Text cleaning and normalization

### 2. **AI Models**
- **LLM**: NousResearch/Hermes-2-Pro-Mistral-7B (4-bit quantized)
- **Embeddings**: BAAI/bge-large-en-v1.5 for semantic similarity
- **Quantization**: BitsAndBytesConfig for memory efficiency

### 3. **Semantic Search & RAG**
- **Vector Embeddings**: Normalized sentence embeddings for semantic similarity
- **Cosine Similarity**: Fast similarity computation for context retrieval
- **Context-Aware Generation**: Retrieved context fed to LLM for accurate responses

### 4. **Comparison Engine**
- **Multi-level Analysis**: Character, word, and line-level similarity metrics
- **Visual Diff**: HTML-based side-by-side comparison with highlighting
- **AI Analysis**: Comprehensive report generation with impact assessment

## 🚀 Installation

### Prerequisites
Ensure you have Python 3.8+ and the following system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install poppler tesseract
```

**Windows:**
- Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Download and install [Poppler utilities](https://blog.alivate.com.au/poppler-windows/)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/pdf-analysis-suite.git
cd pdf-analysis-suite
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download models (automatic on first run):**
The application will automatically download the required models on first startup:
- NousResearch/Hermes-2-Pro-Mistral-7B (~4GB with quantization)
- BAAI/bge-large-en-v1.5 (~1.3GB)

## 📖 Usage

### Starting the Application
```bash
python main.py
```

The application will start a Gradio web interface accessible at `http://localhost:7860`

### PDF Comparison
1. Navigate to the **"Compare PDFs"** tab
2. Upload two PDF files (PDF 1 and PDF 2)
3. Click **"Run Comparison"** to analyze differences
4. View results in:
   - **Similarities & Differences**: AI-generated comprehensive analysis
   - **Visual Diff**: Side-by-side highlighting of changes

### Interactive PDF Chat
1. Navigate to the **"Interact with PDF"** tab
2. Upload a PDF file
3. **Generate Summary**: Click to get an AI-generated document summary
4. **Chat**: Ask questions about the document content
   - Example: "What are the main conclusions?"
   - Example: "Summarize the methodology section"

### Advanced Features
- **Cancellation**: Stop long-running operations using the cancel button
- **Progress Tracking**: Monitor processing status for large documents
- **Memory Management**: Automatic model caching and efficient memory usage

## ⚙️ Configuration

### Model Settings
Edit the configuration variables in `main.py`:

```python
LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Mistral-7B"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Performance Tuning
- **GPU Memory**: Adjust quantization settings for your hardware
- **Chunk Size**: Modify `max_len` parameter in `chunk_text()` function
- **Generation Parameters**: Tune temperature, top_p, and max_tokens

## 📊 Performance Metrics

### Similarity Metrics
- **Character Similarity**: Exact character-level matching
- **Word Similarity**: Unique word overlap analysis
- **Line Similarity**: Structural comparison
- **Content Analysis**: Addition/deletion/modification detection

### Processing Capabilities
- **Document Size**: Handles documents up to several hundred pages
- **Languages**: Primarily optimized for English (OCR supports multiple languages)
- **File Formats**: PDF files (with automatic OCR for image-based content)

## 🛠️ System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended for large documents)
- **Storage**: 10GB free space for models and temporary files
- **CPU**: Multi-core processor recommended
- **Python**: 3.8 or higher

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)
- **RAM**: 16GB or higher
- **Storage**: SSD for faster model loading

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - Pre-trained language models
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic text embeddings
- **[Gradio](https://gradio.app/)** - Web interface framework
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF processing library
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** - Optical character recognition
- **[BitsAndBytesConfig](https://github.com/TimDettmers/bitsandbytes)** - Model quantization
- **[NousResearch](https://huggingface.co/NousResearch)** - Hermes-2-Pro-Mistral-7B model
- **[BAAI](https://huggingface.co/BAAI)** - BGE embedding models

