# LLM Wiki - Local Second Brain Demo

This project is a minimal, dependency-free local demo of Andrej Karpathy's "LLM Wiki" concept. 

It organizes a folder of markdown files into a persistent, interlinked knowledge base ("Wiki") using an LLM. It operates entirely on local or cloud-based LLM APIs without requiring complex vector database or chunking framework setups.

---

## Folder Structure

- `raw/` - Put your source document files here (e.g. `.txt` articles). These remain read-only.
- `wiki/` - The compiled knowledge base created and edited by the LLM. It includes concept pages, source summaries, `index.md`, and `log.md`.
- `main.py` - The CLI script managing all ingest, query, and lint operations (uses only Python standard libraries).
- `CLAUDE.md` - The instruction schema for the LLM.

---

## Prerequisites

Python 3.x installed. Choose one of the following LLM backends:

### Option A: Local Ollama (Recommended, Privacy-First)
1. Install and run [Ollama](https://ollama.com/).
2. Pull a model (e.g., Qwen 2.5 7B or Llama 3):
   ```bash
   ollama pull qwen2.5:7b
   ```

### Option B: Gemini API
Set your API key as an environment variable:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Option C: OpenAI API
Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

## How to Run

Navigate to this directory in your terminal and run the commands below.

### 1. Check Status
Check how many source files are currently registered:
```bash
python3 main.py status
```

### 2. Ingest Raw Sources
Read and compile new sources from `/raw` into `/wiki`:
* **With Ollama (Local)**:
  ```bash
  python3 main.py ingest --provider ollama --model qwen2.5:7b
  ```
* **With Gemini (Cloud)**:
  ```bash
  python3 main.py ingest --provider gemini --model gemini-2.5-flash
  ```
* **With OpenAI (Cloud)**:
  ```bash
  python3 main.py ingest --provider openai --model gpt-4o-mini
  ```

### 3. Query the Wiki
Ask the LLM questions based on your compiled wiki. The script will automatically ask the LLM to identify relevant wiki pages, read only those pages, and generate a synthesized response with citations:
```bash
python3 main.py query -q "What is Model Context Protocol?" --provider gemini
```

### 4. Lint the Wiki
Perform a health check on the knowledge graph (looks for missing metadata frontmatter, dangling links, and orphan pages):
```bash
python3 main.py lint
```

---

## Open in Obsidian
You can open this demo folder (`projects/llm-wiki-demo`) as a Vault in **Obsidian** to browse the created index, log, entity pages, and trace the interlink graph in real time!
