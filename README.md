# Medical RAG System with Qwen3 & FAISS

A specialized Retrieval-Augmented Generation (RAG) system for medical guidelines, featuring section-aware chunking and citation validation.

## 🚀 Features
- **Medical Embeddings**: Powered by `embeddinggemma-300m-medical`.
- **LLM Reasoning**: Uses `Qwen3-8B` (4-bit quantized) for evidence-based answers.
- **Citation Engine**: Automatically validates and maps source identifiers to original document sections and pages.
- **Section-Aware Chunking**: Preserves document structure for better context retrieval.

## 🛠️ Tech Stack
- **Vector DB**: FAISS
- **Models**: Hugging Face Transformers, Sentence-Transformers
- **Framework**: LangChain (Text Splitters)
- **UI**: Gradio

## 📁 Project Structure
- `src/`: Core logic (Processor, Vector Store, Generator, Pipeline).
- `app.py`: Gradio web interface.
- `requirements.txt`: Project dependencies.

## ⚠️ Disclaimer
This project is for educational purposes only. Always consult a qualified healthcare professional for medical advice.
