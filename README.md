# Personal Operating System RAG (POS-RAG)

POS-RAG is a local-first Retrieval-Augmented Generation app that turns your personal files into a mode-based assistant: Study, Build, Career, Life, Health. It answers using your sources, shows citations, and refuses when evidence is weak.

If you want a “second brain” that behaves like software (not vibes), this is it.

---

## Why POS-RAG

Most “chat with PDFs” demos fail in real usage: updates don’t reflect, retrieval is messy, and answers get confident without proof. POS-RAG is built around what makes RAG actually usable:

1) Incremental indexing so edits and new files take effect reliably  
2) Mode-based retrieval so the right knowledge base is searched  
3) Confidence gating so the system refuses instead of hallucinating  
4) Citations so every answer is verifiable  

---

## What it does

### Mode-based assistant (your personal OS)
Documents are organized into intent-based modes:

1) Study: courses, labs, slides, rubrics, lecture notes  
2) Build: engineering notes, troubleshooting logs, design docs  
3) Career: resume versions, job descriptions, interview notes  
4) Life: leases, policies, warranties, admin documents  
5) Health: logs, routines, nutrition notes, symptom tracking  

### Grounded answers with citations
Answers include citations pointing to the source file and (when available) section/page context.

### Refusal rules (anti-hallucination)
If retrieval confidence is low, POS-RAG refuses and asks you to rephrase, select a different mode, or reindex. This prevents confident nonsense when your sources don’t support the answer.

### Local-first
Sources live on your machine. Indexing is local. LLM generation can be local via Ollama.

---

## How it works (high level)

1) You place files into `data/sources/<mode>/`  
2) The indexer chunks documents, creates embeddings, and builds a per-mode FAISS vector index  
3) A query retrieves top matching chunks for the selected mode  
4) If confidence is strong, an LLM generates an answer using only those retrieved chunks  
5) The response includes the answer, citations, and optional debug scores  

---

## Supported file types

1) PDF (`.pdf`)  
2) Word (`.docx`)  
3) Markdown (`.md`, `.markdown`)  
4) Text (`.txt`)  

Note: scanned/image-only PDFs may not extract text well without OCR.

---

## Tech stack

1) Backend: FastAPI  
2) UI: Streamlit  
3) Embeddings: sentence-transformers (local embeddings)  
4) Vector search: FAISS  
5) Metadata store: SQLite  
6) LLM (optional): Ollama (local)  

---

## Quickstart

### 1) Clone the repo
```bash
git clone <YOUR_REPO_URL>
cd pos_rag
```

### 2) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Create mode folders
```bash
mkdir -p data/sources/{study,build,career,life,health}
```

### 5) Add some files
Examples:
```bash
cp ~/Downloads/syllabus.pdf data/sources/study/
cp ~/Downloads/Lab5.docx data/sources/study/
cp ~/Downloads/resume.docx data/sources/career/
```

### 6) Index your sources
Index all modes:
```bash
python -m scripts.reindex --modes study build career life health
```

Index only one mode:
```bash
python -m scripts.reindex --modes study
```

### 7) Start the API
Terminal window 1:
```bash
uvicorn app.main:app --reload --port 8000
```

### 8) Start the UI
Terminal window 2:
```bash
source .venv/bin/activate
streamlit run app/ui_streamlit.py
```

Open the Streamlit URL printed in your terminal (usually `http://localhost:8501`).

---

## Enable local LLM generation (Ollama)

POS-RAG can use a local LLM via Ollama for cleaner answers while keeping citations and refusal rules.

### 1) Install Ollama
Install via Homebrew if available on your machine:
```bash
brew install ollama
```

### 2) Start Ollama
```bash
ollama serve
```

If you see “address already in use”, Ollama is already running.

### 3) Pull a model
```bash
ollama pull llama3.1:8b
```

### 4) Verify Ollama is reachable
```bash
curl http://127.0.0.1:11434/api/tags
```

---

## Use from the terminal (API)

### Status
```bash
curl http://127.0.0.1:8000/status
```

### Reindex
All modes:
```bash
curl -X POST http://127.0.0.1:8000/reindex \
  -H "Content-Type: application/json" \
  -d '{"modes": null}'
```

Single mode:
```bash
curl -X POST http://127.0.0.1:8000/reindex \
  -H "Content-Type: application/json" \
  -d '{"modes": ["study"]}'
```

### Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"mode":"study","question":"Summarize Lab 5 requirements and list deliverables.","debug":true}' | python -m json.tool
```

---

## Tips for better answers

1) Choose the right mode for the question  
2) Ask specific questions (vague prompts reduce retrieval confidence)  
3) Reindex after adding or editing documents  
4) Turn on Debug in the UI to inspect similarity scores and citations  

---

## Repo structure

```text
pos_rag/
  app/
    main.py
    config.py
    db.py
    ui_streamlit.py
    ingest/
    retrieval/
    llm/
  scripts/
  data/
    sources/
      study/
      build/
      career/
      life/
      health/
    index/
    sqlite/
```

---

## Roadmap ideas

1) Hybrid search (BM25 + vectors) for stronger keyword matching  
2) Reranking for higher precision retrieval  
3) File watcher for automatic incremental reindexing  
4) Better citation UX (click-to-open, page jumping)  
5) Evaluation suite (golden question set + citation checks)  

---

## License

See `LICENSE` for details.

---

## Acknowledgements

Built with FastAPI, Streamlit, FAISS, SQLite, sentence-transformers, and Ollama.
