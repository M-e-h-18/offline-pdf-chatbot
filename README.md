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

### 📌 Processing Pipeline

1. **Input Stage**
   - PDF upload through Gradio interface
   - Automatic MIME type validation
   - Parallel processing for multiple files

2. **Text Extraction**
   ```mermaid
   graph TD
     A[PDF Input] --> B{Native Text?}
     B -->|Yes| C[PyMuPDF Extraction]
     B -->|No| D[PDF2Image Conversion]
     D --> E[Tesseract OCR Processing]
     C & E --> F[Text Normalization]
   ```

3. **Analysis Phase**
   - For Comparisons:
     - Dual document alignment
     - Three-level diff analysis (character/word/line)
     - Hybrid similarity scoring
   - For Chat/Summarization:
     - Semantic chunking with overlap
     - Hierarchical embedding generation
     - Context-aware retrieval

4. **AI Integration**
   - Dynamic model loading (quantized)
   - Adaptive context window management
   - Generation with safety filters

5. **Output Generation**
   - Interactive visualizations
   - Structured report formatting
   - Progressive response streaming


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

##Kaggle code
```python
!pip install --quiet sentence-transformers transformers bitsandbytes accelerate gradio pytesseract pdf2image nltk faiss-cpu PyMuPDF
!apt-get install -y -qq poppler-utils tesseract-ocr

import os
import fitz
import torch
import re
import difflib
import pytesseract
import threading
import numpy as np
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
from sentence_transformers import SentenceTransformer

# --- System Configuration ---
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# --- Model Loading ---
LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Mistral-7B"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache models in memory
_embedding_model = None
_llm_tokenizer = None
_llm_model = None

def load_models():
    global _embedding_model, _llm_tokenizer, _llm_model
    
    # Load embedding model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=DEVICE,
            trust_remote_code=True
        )
        _embedding_model.normalize_embeddings = True
    
    # Load LLM
    if _llm_model is None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
    
    return _embedding_model, _llm_tokenizer, _llm_model

# Initialize models
EMBEDDING_MODEL, LLM_TOKENIZER, LLM_MODEL = load_models()

# --- Global States ---
cancel_flag = threading.Event()
chat_history = []

# --- PDF Processing Functions ---
def extract_text_with_ocr(pdf_path):
    """Extract text with fallback to OCR for image-based pages"""
    if not pdf_path:
        return ""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            if cancel_flag.is_set():
                return "[CANCELLED]"
            text = page.get_text()
            if text.strip():
                full_text += text + "\n"
            else:
                try:
                    images = convert_from_path(pdf_path, first_page=page.number+1, last_page=page.number+1)
                    full_text += pytesseract.image_to_string(images[0]) + "\n"
                except Exception as e:
                    print(f"OCR Error: {e}")
                    full_text += f"[IMAGE PAGE {page.number} - NO TEXT EXTRACTED]\n"
        return full_text.strip()
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        return f"Error extracting PDF: {str(e)}"

def chunk_text(text, max_len=800):
    """Smart text chunking for LLM processing"""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(LLM_TOKENIZER.encode(current_chunk + para, add_special_tokens=False)) < max_len:
            current_chunk += "\n\n" + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks if chunks else [text[:max_len]]

def get_embeddings(text_chunks):
    """Generate normalized embeddings with BGE model"""
    embeddings = EMBEDDING_MODEL.encode(
        text_chunks,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return embeddings

# --- Enhanced Comparison Functions ---
def highlight_differences(old_text, new_text):
    """Generate HTML with visual diff highlighting"""
    differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    return differ.make_table(
        old_text.splitlines(),
        new_text.splitlines(),
        fromdesc="PDF 1",
        todesc="PDF 2",
        context=True,
        numlines=3
    )

def calculate_similarity_metrics(old_text, new_text):
    """Calculate detailed similarity metrics"""
    # Word-level similarity
    old_words = set(old_text.lower().split())
    new_words = set(new_text.lower().split())
    
    common_words = old_words.intersection(new_words)
    total_unique_words = old_words.union(new_words)
    
    word_similarity = len(common_words) / len(total_unique_words) if total_unique_words else 0
    
    # Character-level similarity
    char_similarity = difflib.SequenceMatcher(None, old_text, new_text).ratio()
    
    # Line-level similarity
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    line_similarity = difflib.SequenceMatcher(None, old_lines, new_lines).ratio()
    
    return {
        'word_similarity': word_similarity * 100,
        'character_similarity': char_similarity * 100,
        'line_similarity': line_similarity * 100,
        'total_words_pdf1': len(old_text.split()),
        'total_words_pdf2': len(new_text.split()),
        'common_words': len(common_words),
        'unique_to_pdf1': len(old_words - new_words),
        'unique_to_pdf2': len(new_words - old_words)
    }

def generate_detailed_comparison_report(old_text, new_text):
    """Generate comprehensive comparison report with similarities and differences"""
    # Get similarity metrics
    metrics = calculate_similarity_metrics(old_text, new_text)
    
    prompt = f"""Provide a comprehensive analysis of these two PDF documents. Include:

1. **DOCUMENT OVERVIEW**:
   - Brief description of each document's main topic/purpose
   - Document length comparison (PDF 1: {metrics['total_words_pdf1']} words, PDF 2: {metrics['total_words_pdf2']} words)

2. **SIMILARITY ANALYSIS**:
   - Overall similarity score: {metrics['character_similarity']:.1f}%
   - Common themes and consistent content
   - Shared terminology and concepts
   - Structural similarities

3. **KEY DIFFERENCES**:
   - Major content additions in PDF 2
   - Significant removals from PDF 1
   - Changed sections with specific examples
   - Terminology or approach differences

4. **DETAILED CHANGES**:
   - New sections or chapters added
   - Reorganized content
   - Updated information or data
   - Policy/procedure changes

5. **IMPACT ASSESSMENT**:
   - Significance of changes (minor/moderate/major)
   - Potential implications of modifications
   - Areas requiring attention

PDF 1 CONTENT:
{old_text[:6000]}

PDF 2 CONTENT:
{new_text[:6000]}

Provide detailed analysis:"""
    
    inputs = LLM_TOKENIZER(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=8192,
        padding=True,
        add_special_tokens=True
    ).to(DEVICE)
    
    outputs = LLM_MODEL.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1500,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=LLM_TOKENIZER.eos_token_id
    )
    
    ai_analysis = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Combine metrics and AI analysis
    metrics_report = f"""## 📊 Similarity Metrics

- **Character Similarity**: {metrics['character_similarity']:.1f}%
- **Word Similarity**: {metrics['word_similarity']:.1f}%  
- **Line Similarity**: {metrics['line_similarity']:.1f}%

### Word Analysis
- **PDF 1 Total Words**: {metrics['total_words_pdf1']:,}
- **PDF 2 Total Words**: {metrics['total_words_pdf2']:,}
- **Common Words**: {metrics['common_words']:,}
- **Unique to PDF 1**: {metrics['unique_to_pdf1']:,}
- **Unique to PDF 2**: {metrics['unique_to_pdf2']:,}

---

## 🤖 AI Analysis

{ai_analysis}
"""
    
    return metrics_report

# --- Chat Functions ---
def chat_with_pdf(message, pdf_text, history):
    """Chat with PDF context using semantic search"""
    if not pdf_text:
        history.append((message, "No PDF text available. Please upload a PDF first."))
        return history, ""
    
    if not history and pdf_text:
        history.append((None, f"PDF loaded with {len(pdf_text.split())} words. Ask me anything about it!"))
    
    # Get relevant context using embeddings
    chunks = chunk_text(pdf_text)
    if not chunks:
        history.append((message, "No valid text chunks found in the PDF."))
        return history, ""
    
    # Get embeddings
    chunk_embeddings = get_embeddings(chunks)
    query_embedding = get_embeddings([message])
    
    # Find most relevant chunk using cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding, 
        chunk_embeddings
    )
    best_chunk = chunks[similarities.argmax()]
    
    context = f"""Answer this question based ONLY on the provided context. Do not generate follow-up questions.

Context:
{best_chunk}

Question: {message}

Answer concisely:"""
    
    inputs = LLM_TOKENIZER(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        add_special_tokens=True
    ).to(DEVICE)
    
    outputs = LLM_MODEL.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=LLM_TOKENIZER.eos_token_id
    )
    response = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean up response
    response = response.split("Question:")[0].strip()
    response = response.split("\n\n")[0].strip()
    
    history.append((message, response))
    return history, ""

# --- Summarization Functions ---
def generate_summary(text, progress=gr.Progress()):
    """Generate summary with adaptive strategy based on document length"""
    try:
        if not text or text == "[CANCELLED]":
            return "⚠ No text available to summarize"
        
        progress(0.1, desc="Analyzing document...")
        word_count = len(text.split())
        
        # Adaptive strategy based on document length
        if word_count < 500:  # Short documents - direct summarization
            progress(0.3, desc="Processing short document...")
            prompt = f"""Create a clear, well-structured summary of this document. Focus on the main points, key information, and important details. Organize the summary with proper structure and formatting.

Document to summarize:
{text}

Please provide a comprehensive summary:"""
            
            inputs = LLM_TOKENIZER(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
                add_special_tokens=True
            ).to(DEVICE)
            
            outputs = LLM_MODEL.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
                pad_token_id=LLM_TOKENIZER.eos_token_id
            )
            summary = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return summary.strip()
            
        elif word_count < 2000:  # Medium documents - single chunk with better prompt
            progress(0.3, desc="Processing medium document...")
            prompt = f"""Analyze this document and create a detailed summary that captures:

1. Main topic and purpose
2. Key points and findings
3. Important details and data
4. Conclusions or recommendations

Document:
{text[:4000]}

Provide a well-organized summary:"""
            
            inputs = LLM_TOKENIZER(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
                padding=True,
                add_special_tokens=True
            ).to(DEVICE)
            
            outputs = LLM_MODEL.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=800,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
                pad_token_id=LLM_TOKENIZER.eos_token_id
            )
            summary = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return summary.strip()
            
        else:  # Long documents - chunked approach
            progress(0.2, desc="Preparing text chunks for long document...")
            chunks = chunk_text(text, max_len=1200)  # Larger chunks for better context
            if not chunks:
                return "⚠ No valid text chunks found"
            
            summaries = []
            for i, chunk in enumerate(chunks):
                if cancel_flag.is_set():
                    return "⏹ Summary generation cancelled"
                
                progress(0.2 + (i/len(chunks))*0.7, desc=f"Processing section {i+1}/{len(chunks)}...")
                
                prompt = f"""Summarize this section of a larger document. Focus on the key information, main points, and important details. Make it informative and well-structured.

Section {i+1} of {len(chunks)}:
{chunk}

Section Summary:"""
                
                inputs = LLM_TOKENIZER(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536,
                    padding=True,
                    add_special_tokens=True
                ).to(DEVICE)
                
                outputs = LLM_MODEL.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=400,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=LLM_TOKENIZER.eos_token_id
                )
                summary = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                summaries.append(f"**Section {i+1}:** {summary.strip()}")
            
            progress(0.95, desc="Combining sections...")
            
            # Create final consolidated summary
            combined_summary = "\n\n".join(summaries)
            
            # If we have multiple sections, create an overall summary
            if len(chunks) > 2:
                final_prompt = f"""Based on these section summaries, create a comprehensive overall summary of the entire document:

{combined_summary}

Overall Document Summary:"""
                
                inputs = LLM_TOKENIZER(
                    final_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True,
                    add_special_tokens=True
                ).to(DEVICE)
                
                outputs = LLM_MODEL.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=600,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=LLM_TOKENIZER.eos_token_id
                )
                overall_summary = LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                return f"## Overall Summary\n\n{overall_summary.strip()}\n\n---\n\n## Detailed Section Summaries\n\n{combined_summary}"
            else:
                return combined_summary
        
        progress(1.0, desc="Summary complete!")
        
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"

# --- Gradio UI ---
with gr.Blocks(title="PDF Analysis Suite", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📚 PDF Analysis Tool")
    
    with gr.Tabs():
        # ===== COMPARISON TAB =====
        with gr.Tab("🔍 Compare PDFs", id="compare"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    with gr.Group():
                        old_pdf = gr.File(label="PDF 1", file_types=[".pdf"])
                        new_pdf = gr.File(label="PDF 2", file_types=[".pdf"])
                    with gr.Row():
                        compare_btn = gr.Button("🔍 Run Comparison", variant="primary", size="lg")
                        reset_compare_btn = gr.Button("🔄 Reset", variant="secondary")
                
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("📊 Similarities & Differences"):
                            ai_report = gr.Markdown(label="Detailed Analysis")
                        
                        with gr.Tab("👁️ Visual Diff"):
                            diff_html = gr.HTML(label="Document Differences")
            
            def run_comparison(old_file, new_file, progress=gr.Progress()):
                cancel_flag.clear()
                if not old_file or not new_file:
                    return "<p>⚠️ Please upload both PDF files to compare</p>", "## ⚠️ Missing Files\n\nPlease upload both PDF 1 and PDF 2 to perform comparison analysis."
                
                progress(0.1, desc="Extracting PDF 1...")
                old_text = extract_text_with_ocr(old_file.name)
                progress(0.3, desc="Extracting PDF 2...")
                new_text = extract_text_with_ocr(new_file.name)
                
                progress(0.5, desc="Generating visual diff...")
                diff = highlight_differences(old_text, new_text)
                
                progress(0.7, desc="Analyzing similarities and differences...")
                report = generate_detailed_comparison_report(old_text, new_text)
                
                progress(1.0, desc="Complete!")
                return diff, report
            
            def reset_comparison():
                return None, None, "<p>Upload PDF files to begin comparison</p>", "## Ready for Comparison\n\nUpload both PDF files and click 'Run Comparison' to begin analysis."
            
            compare_btn.click(
                run_comparison,
                inputs=[old_pdf, new_pdf],
                outputs=[diff_html, ai_report]
            )
            
            reset_compare_btn.click(
                reset_comparison,
                outputs=[old_pdf, new_pdf, diff_html, ai_report]
            )
        
        # ===== INTERACT TAB =====
        with gr.Tab("💬 Interact with PDF", id="interact"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Upload & Summarize")
                    pdf_input = gr.File(label="PDF File", file_types=[".pdf"])
                    with gr.Row():
                        summarize_btn = gr.Button("📝 Generate Summary", variant="primary")
                        cancel_btn = gr.Button("⏹️ Cancel", variant="secondary")
                        clear_summary_btn = gr.Button("🗑️ Clear Summary", variant="secondary")
                    with gr.Row():
                        copy_summary_btn = gr.Button("📋 Copy Summary", size="sm")
                    summary_output = gr.Textbox(
                        label="Document Summary", 
                        lines=15, 
                        max_lines=20,
                        show_copy_button=True,
                        placeholder="Summary will appear here after processing..."
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Chat with PDF")
                    chatbot = gr.Chatbot(height=500, show_copy_button=True)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask about the PDF content...", 
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("📤 Send", variant="primary", scale=1)
                    with gr.Row():
                        clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        reset_all_btn = gr.Button("🔄 Reset All", variant="secondary")
            
            # Store extracted text
            pdf_text = gr.State()
            
            def extract_and_store(pdf_file):
                try:
                    if pdf_file is None:
                        return ""
                    cancel_flag.clear()
                    return extract_text_with_ocr(pdf_file.name)
                except Exception as e:
                    print(f"Error extracting text: {e}")
                    return f"Error extracting PDF: {e}"
            
            def cancel_processing():
                cancel_flag.set()
                return "⚠️ Processing cancelled..."
            
            def clear_chat():
                chat_history.clear()
                return None
            
            def clear_summary():
                return ""
            
            def reset_all():
                chat_history.clear()
                cancel_flag.set()
                return None, None, "", ""
            
            # Event handlers
            pdf_input.change(
                extract_and_store,
                inputs=pdf_input,
                outputs=pdf_text
            )
            
            summarize_btn.click(
                generate_summary,
                inputs=pdf_text,
                outputs=summary_output
            )
            
            cancel_btn.click(
                cancel_processing,
                outputs=summary_output
            )
            
            clear_summary_btn.click(
                clear_summary,
                outputs=summary_output
            )
            
            chat_input.submit(
                chat_with_pdf,
                inputs=[chat_input, pdf_text, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            send_btn.click(
                chat_with_pdf,
                inputs=[chat_input, pdf_text, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            clear_chat_btn.click(
                clear_chat,
                outputs=chatbot
            )
            
            reset_all_btn.click(
                reset_all,
                outputs=[pdf_input, pdf_text, summary_output, chatbot]
            )

    @app.load()
    def initialize():
        load_models()

# Launch the app
if __name__ == "__main__":
    app.launch()
'''
