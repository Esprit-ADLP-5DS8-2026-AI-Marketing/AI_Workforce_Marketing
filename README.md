# RAG for cosmetic marketing

This branch contains the document search and question-answering part of our
AI Workforce Marketing project at ESPRIT.

The aim is to give the campaign agents access to product information, brand
guidelines and advertising rules. The current version runs from the terminal.
It retrieves relevant passages and uses Gemini to answer questions with sources.

## How it works

The preparation script extracts text from four PDFs and keeps the source and
page numbers. We compare two ways of splitting that text: fixed token windows
and document sections.

We use `intfloat/multilingual-e5-small` to create 384-dimensional embeddings and
store them in a local Qdrant database. For each question, the system retrieves
four passages by default, sends them to Gemini and checks the returned citations.
It can report that the retrieved passages do not contain enough information.

An accepted citation means its excerpt exists in the retrieved text. It does
not guarantee that every claim in the answer is correct.

## Documents

Place the four source PDFs in `data/raw/`:

| File | Purpose |
| --- | --- |
| `Lumera_Aqua_Lumina_Brand_Product_Book.pdf` | Product details and brand guidelines for a fictional demonstration brand. |
| `Standards publicitaires meta.pdf` | An export of Meta advertising standards. |
| `Recommandation-Produits-cosmetiques-v8.pdf` | French professional guidance on cosmetic advertising. |
| `rag_database_content.pdf` | Internal writing methods, visual guidance and review checklists. |

The ingestion rules currently target these four documents. Adding another PDF
requires updating the preparation code. Lumera's prices, claims and test results
are fictional and must not be treated as evidence for a real product.

## Setup

Use Python 3.10 or newer. The first run downloads the tokenizer and embedding
model. Gemini requires an API key and an internet connection.

```bash
git clone --branch RAG https://github.com/Esprit-ADLP-5DS8-2026-AI-Marketing/AI_Workforce_Marketing.git
cd AI_Workforce_Marketing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rag.txt
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead.

Edit `.env` locally and set `GEMINI_API_KEY` and `GEMINI_MODEL` to a model available
in your Google project. Never commit the real key. The dependency file covers
this RAG module; it is not a lockfile for the team's full development environment.

## Prepare the data

Run these commands from the repository root:

```bash
python scripts/prepare_corpus.py
python scripts/chunk_corpus.py
python scripts/build_indexes.py
python scripts/evaluate_retrieval.py
```

The initial corpus produces 65 document units, then 79 fixed-window chunks or
144 section-based chunks. Both collections are built for comparison. Use
`--rebuild` with `build_indexes.py` after changing the chunks if the collections
already exist.

## Ask a question

Inspect the retrieved passages without calling Gemini:

```bash
python scripts/search_rag.py "Quelle est la concentration de niacinamide dans Aqua Lumina ?" --strategy sections
```

Get an answer with citations:

```bash
python scripts/answer_rag.py "Quelle est la concentration de niacinamide dans Aqua Lumina ?"
```

The output includes `status`, `answer`, `citations` and `limitations`. Run traces
are saved locally in `data/answers/`.

Useful options for `answer_rag.py`:

- `--dry-run` prepares the context without calling Gemini.
- `--offline` uses cached E5 files. Gemini still needs internet.
- `--strategy fixed` selects the other chunking strategy.
- `--model MODEL_ID` overrides the configured Gemini model.
- `--api-test` makes a minimal API call without loading E5 or Qdrant.

## Project files

| Path | Contents |
| --- | --- |
| `scripts/prepare_corpus.py` | PDF extraction, cleaning and source metadata. |
| `scripts/chunk_corpus.py` | The two chunking strategies and coverage checks. |
| `scripts/rag_common.py` | Shared E5 and Qdrant functions. |
| `scripts/build_indexes.py` | Embedding calculation and index creation. |
| `scripts/evaluate_retrieval.py` | Retrieval comparison on annotated questions. |
| `scripts/search_rag.py` | Passage search without generation. |
| `scripts/answer_rag.py` | Gemini calls, answer validation and run traces. |
| `scripts/repair_scipy*.py` | Maintenance tools used for a specific macOS SciPy import issue. |
| `prompts/` | Current and earlier answer prompts. |
| `tests/` | Tests for chunking, retrieval and answer handling. |
| `data/raw/` | Source PDFs. |
| `data/evaluation/questions.jsonl` | Questions and expected evidence used for retrieval evaluation. |

Processed text, chunks, embeddings, Qdrant storage and run results are generated
locally and excluded from Git. API keys and virtual environments are excluded too.

## Tests and current limits

After preparing the corpus and downloading the tokenizer, run:

```bash
python tests/test_chunking.py
python tests/test_retrieval.py
python tests/test_answer_rag.py
```

The Gemini transport is mocked in the answer tests. The retrieval evaluation
uses 14 questions, including two without a supplied answer. This is a small
academic evaluation, not a general guarantee of answer quality or ad compliance.

The current version is text-only. It has no reranker, HTTP API, Docker image or
n8n integration yet. Local Qdrant storage should not be opened by several
processes at once. These points will be addressed before connecting the module
to the team's agents.
