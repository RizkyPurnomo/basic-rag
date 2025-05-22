# Retrieval-Augmented Generation (RAG) System
A modular and local-first Retrieval-Augmented Generation (RAG) system built with Ollama and ChromaDB. This project retrieves relevant text chunks from local vector storage and uses a local language model to generate answers grounded in the retrieved context. It also includes an automated grading system to evaluate the quality of generated answers using a separate local model.

---

## Project Structure
<pre>
PROJECT
│
├── data/ # Raw data and local ChromaDB storage
│ ├── raw/ # Original files (PDFs, text, etc.)
│ └── chromadb/ # Persistent ChromaDB storage
│
├── notebooks/ # Jupyter notebooks for experimentation
│
├── src/ # Source code
│ ├── data_builder.py # Data preparation and chunking logic
│ ├── prompts.py # Prompt templates for generation and grading
│ ├── models.py # RAG logic using LLMs (API or remote)
│ ├── local_models.py # Local RAG and grading using Ollama
│ └── questions.py # List of questions
│
├── config.env # Environment variables
└── requirements.txt # Python dependencies
</pre>
## Features
- **RAG with Local Models**: Use `llama3.1:8b` for generation and `deepseek-r1:1.5b` for grading with [Ollama](https://ollama.com/).
- **Vector Search**: Fast retrieval with [ChromaDB](https://www.trychroma.com/).
- **Automated Grading**: Using LLM-as-a-Judge paradigm to score and explain outputs using metrics such as correctness, groundedness, and clarity.
- **Modular Design**: Clear separation of concerns across multiple Python files.
