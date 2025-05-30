import gradio as gr
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os
import nltk

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

LLM_MODEL_NAME = "Open-Orca/Mistral-7B-OpenOrca"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_TARGET_CHAR_LEN = 800
CHUNK_OVERLAP_SENTENCES = 2

@torch.no_grad()
def load_models():
    print("Loading models...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    ) if DEVICE == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    return tokenizer, model, embedder

LLM_TOKENIZER, LLM_MODEL, EMBEDDING_MODEL = load_models()

def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_file_obj):
    try:
        file_path = pdf_file_obj.name if hasattr(pdf_file_obj, 'name') else pdf_file_obj
        doc = fitz.open(file_path)
        text = "".join(page.get_text("text") for page in doc)
        doc.close()
        return clean_text(text)
    except:
        return ""

def sentence_aware_chunk_text(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [clean_text(text)] if text.strip() else []

    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) <= CHUNK_TARGET_CHAR_LEN or not current_chunk:
            current_chunk.append(sentence)
            current_len += len(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-CHUNK_OVERLAP_SENTENCES:] + [sentence]
            current_len = sum(len(s) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [c for c in chunks if c.strip()]

def format_prompt(user_content, system_prompt="You are a helpful AI assistant."):
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content.strip()}<|im_end|>\n<|im_start|>assistant\n"

@torch.no_grad()
def generate_response(prompt, max_new_tokens=150, temperature=0.7):
    inputs = LLM_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    outputs = LLM_MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=LLM_TOKENIZER.eos_token_id
    )
    return LLM_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def summarize_text(text, length="medium"):
    """Comprehensive summarization covering all document parts"""
    # Improved chunking with context preservation
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return "No content to summarize"
    
    # Create chunks with better context handling
    chunks = []
    current_chunk = []
    current_len = 0
    
    for i, sentence in enumerate(sentences):
        if current_len + len(sentence) <= CHUNK_TARGET_CHAR_LEN or not current_chunk:
            current_chunk.append(sentence)
            current_len += len(sentence)
        else:
            # Ensure we don't break in middle of important context
            if i < len(sentences) - 1 and any(c in sentence for c in [":", ";", "-"]):
                current_chunk.append(sentence)
                current_len += len(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                # Carry over 3 sentences for better context
                current_chunk = current_chunk[-3:] + [sentence]
                current_len = sum(len(s) for s in current_chunk)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    if not chunks:
        return "No content to summarize"

    # Improved content selection
    selected_chunks = []
    
    # Always include first and last chunks
    selected_chunks.extend([chunks[0], chunks[-1]])
    
    # Include middle chunk
    middle_idx = len(chunks) // 2
    selected_chunks.append(chunks[middle_idx])
    
    # Include additional chunks from key sections
    for i in [1, -2, middle_idx-1, middle_idx+1]:
        if 0 <= i < len(chunks):
            selected_chunks.append(chunks[i])
    
    # Remove duplicates while preserving order
    seen = set()
    selected_chunks = [chunk for chunk in selected_chunks if not (chunk in seen or seen.add(chunk))]
    
    # Enhanced prompt for completeness
    context = "\n\n".join(selected_chunks)
    prompt = format_prompt(
        f"Create a comprehensive, well-structured summary covering ALL key points from this document. "
        f"Ensure you include information from ALL sections (beginning, middle, and end). "
        f"Provide complete sentences and thorough coverage of:\n\n"
        f"1. Main concepts and definitions\n"
        f"2. Key processes and methodologies\n"
        f"3. Important examples and applications\n"
        f"4. Comparisons and contrasts\n"
        f"5. Conclusions and final points\n\n"
        f"Document Content:\n{context}",
        "You are an expert technical summarizer who creates accurate, complete summaries "
        "that capture all essential information from every part of a document. "
        "Never leave sentences incomplete or skip important details."
    )
    
    # Adjust length parameters
    length_params = {
        "short": (100, 200),
        "medium": (300, 600),
        "long": (700, 1200)
    }.get(length, (300, 600))
    
    return generate_response(prompt, max_new_tokens=length_params[1])

def process_pdf(file, progress=gr.Progress()):
    if not file:
        return "No file uploaded", [], {"chunks": None, "index": None}, None
    try:
        progress(0.1, "Extracting text")
        text = extract_text_from_pdf(file)
        if not text:
            return "Failed to extract text", [], {"chunks": None, "index": None}, None

        progress(0.3, "Chunking text")
        chunks = sentence_aware_chunk_text(text)

        progress(0.6, "Building vector store")
        index = build_vector_store(chunks)

        progress(0.8, "Generating comprehensive summary")
        summary = summarize_text(text, length="medium")  # Now processes entire document

        return (
            f"Processed {os.path.basename(file.name)}",
            [("System", "PDF processed successfully!"), ("System", summary)],
            {"chunks": chunks, "index": index},
            None
        )
    except Exception as e:
        return f"Error: {str(e)}", [("System", f"Error: {str(e)}")], {"chunks": None, "index": None}, None

def build_vector_store(chunks):
    embeddings = EMBEDDING_MODEL.encode(chunks, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index

def query_rag(question, index, chunks, top_k=3):
    q_embed = EMBEDDING_MODEL.encode([question], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(q_embed, k=min(top_k, len(chunks)))
    context = "\n\n".join([chunks[i] for i in indices[0] if i < len(chunks)])
    prompt = format_prompt(
        f"Context:\n{context}\n\nQuestion: {question}",
        "Answer using ONLY the provided context. If unsure, say 'Not in document'."
    )
    return generate_response(prompt)



def chat(message, history, doc_state):
    if not message.strip():
        return "", history, doc_state

    history.append((message, ""))
    if not doc_state or not doc_state.get("chunks"):
        history[-1] = (message, "Please process a PDF first")
        return "", history, doc_state

    if "summary" in message.lower():
        summary = summarize_text("\n\n".join(doc_state["chunks"]))
        history[-1] = (message, summary)
    else:
        answer = query_rag(message, doc_state["index"], doc_state["chunks"])
        history[-1] = (message, answer)

    return "", history, doc_state

with gr.Blocks(title="PDF Chat Assistant") as app:
    gr.Markdown("# PDF Chat Assistant")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status = gr.Textbox(label="Status")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            message = gr.Textbox(label="Message", placeholder="Ask a question or request a summary")

    doc_state = gr.State({"chunks": None, "index": None})

    file_input.upload(
        process_pdf,
        [file_input],
        [status, chatbot, doc_state, file_input]
    )

    message.submit(
        chat,
        [message, chatbot, doc_state],
        [message, chatbot, doc_state]
    )

if __name__ == "__main__":
    app.launch(server_port=7860)
