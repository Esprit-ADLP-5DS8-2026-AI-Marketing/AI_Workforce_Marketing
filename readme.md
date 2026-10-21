# RAG for cosmetic marketing

This branch contains the document search and question-answering part of our
AI Workforce Marketing project at ESPRIT. It gives the campaign agents access
to product information, brand guidelines, advertising rules and internal
marketing methods.

We started with a naive RAG that returned the four closest passages. The current
version adds reranking, an evaluation set split into development and test,
private product PDF ingestion, a FastAPI service and n8n workflows. The selected
retrieval configuration uses document sections, 20 candidates and up to 8 final
passages.

## How it works

We prepare the documents before answering questions:

1. Extract the PDF text with PyMuPDF, clean repeated elements and keep provenance.
2. Split the selected text into chunks, using either fixed windows or sections.
3. Encode the chunks with `intfloat/multilingual-e5-small` and store their vectors
   and metadata in local Qdrant collections.

For a question, we use the same E5 model with the `query:` prefix. Qdrant retrieves
up to 20 candidate passages by cosine similarity. A multilingual cross-encoder
then scores the question together with each passage, document title and section.
We keep the best 8 passages after reranking.

There are two outputs. Retrieval returns the passages and their sources without
calling Gemini. Answering also calls Gemini and checks the returned JSON and
citations. The Manager's RAG tool uses retrieval mode because it already has
its own language model.

| Component | Configuration used |
| --- | --- |
| Embeddings | `intfloat/multilingual-e5-small`, 384 normalized dimensions |
| Vector search | Local Qdrant, cosine similarity |
| Chunk size | Up to 300 E5 tokens, with a target overlap of 50 tokens |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, local CPU inference |
| Reranker input | Question paired with the document title, section and passage |
| Retrieval budget | 20 candidates, up to 8 final passages |
| Generation | Gemini, selected through `GEMINI_MODEL` |
| Answer prompt | `rag_answer_v1_3` |

E5 and the reranker use pretrained weights. We changed the pipeline, parameters
and prompts, without fine-tuning model weights. Similarity and reranker scores
are ranking signals, not probabilities that a fact is true.

## Source documents

The initial corpus uses four PDFs in `data/raw/`:

| File | Purpose |
| --- | --- |
| `Lumera_Aqua_Lumina_Brand_Product_Book.pdf` | Product and brand information for a fictional demonstration product. |
| `Standards publicitaires meta.pdf` | A PDF snapshot of Meta advertising standards. |
| `Recommandation-Produits-cosmetiques-v8.pdf` | French professional guidance on cosmetic advertising. |
| `rag_database_content.pdf` | Internal writing methods, visual guidance and review checklists. |

Preparation keeps 65 units from 61 raw pages: 9 Lumera pages, 23 Meta pages,
11 ARPP pages and 22 internal records. Chunking then produces 79 fixed-window
chunks or 144 section-based chunks. Units and chunks are different stages.

The initial preparation rules target these four files. New user product PDFs
use the separate ingestion pipeline described below. Lumera's prices, claims
and test results are fictional. Internal advice is not an official policy, and
the ARPP reference has a French scope. Policy answers refer to the stored PDF
versions, without checking the current websites.

## Project files

Commands in this README run from the directory containing `scripts/`,
`prompts/`, `tests/` and `requirements-rag.txt`.

| Path | Role |
| --- | --- |
| `scripts/prepare_corpus.py` | Extract, clean and select the initial corpus. |
| `scripts/chunk_corpus.py` | Build both chunking strategies and check text coverage. |
| `scripts/build_indexes.py` | Create the E5 embeddings and Qdrant collections. |
| `scripts/rag_common.py` | Shared encoding and index functions. |
| `scripts/search_rag.py`, `scripts/rerank_rag.py` | Search, rerank and keep ranking diagnostics. |
| `scripts/answer_rag.py` | Call Gemini, validate citations and save answer traces. |
| `scripts/validate_evaluation.py`, `scripts/evaluate_retrieval.py` | Check annotations and compare retrieval configurations. |
| `scripts/product_rag.py` | Ingest private product PDFs and search their active versions. |
| `scripts/rag_service.py`, `scripts/rag_api.py` | Internal service, HTTP routes and request validation. |
| `scripts/run_rag_api.py`, `scripts/call_rag_api.py` | Start and test the API locally. |
| `scripts/local_qdrant.py` | Manage repeated local-client closes without disabling SQLite thread checks. |
| `prompts/` | Current and earlier answer prompts. |
| `data/evaluation/v2/` | Development/test questions and their version manifest. |
| `tests/` | Tests for retrieval, generation handling, ingestion and the API. |

Processed text, chunks, embeddings, Qdrant databases and run traces are generated
locally. Private product files, credentials and virtual environments must stay
outside Git.

## Setup

Use Python 3.10 or newer. For a fresh checkout:

```bash
git clone --branch RAG https://github.com/Esprit-ADLP-5DS8-2026-AI-Marketing/AI_Workforce_Marketing.git
cd AI_Workforce_Marketing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rag.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` instead.
An existing `ai_env` can also be used. The RAG requirements file is separate
from the team's full environment and is not a complete dependency lockfile.

Create `.env` locally and set `GEMINI_API_KEY` and `GEMINI_MODEL` to a model
available in your Google project. Our recorded generation tests used
`gemini-3.5-flash-lite`. Retrieval and PDF indexing do not require a Gemini key.

The first use downloads the required Hugging Face models. `--offline` reuses
cached model files; it does not make Gemini work without an internet connection.

## Build the initial corpus and indexes

```bash
python scripts/prepare_corpus.py
python scripts/chunk_corpus.py
python scripts/build_indexes.py
```

Check `data/processed/preview/` and the preparation report, especially the
ingredient, price and advertising-rule passages. Source pages remain available
when the linear extraction of a PDF table needs checking.

Both collections are built for comparison. After changing the chunks, rebuild
the existing indexes explicitly:

```bash
python scripts/build_indexes.py --rebuild
```

Stop the API before rebuilding or opening the same local stores from a CLI.

## Search and answer from the terminal

Retrieve and rerank passages without calling Gemini:

```bash
python scripts/search_rag.py "Quelle est la concentration de niacinamide dans Aqua Lumina ?" \
  --strategy sections --rerank --candidate-k 20 --top-k 8
```

Generate an answer using the same retrieval settings:

```bash
python scripts/answer_rag.py "Quelle est la concentration de niacinamide dans Aqua Lumina ?" \
  --strategy sections --rerank --candidate-k 20 --top-k 8
```

These CLI scripts keep a default `top_k` of 4 and require `--rerank` to enable
reranking. The commands above explicitly reproduce our selected configuration.
The API uses sections, reranking, `candidate_k=20` and `top_k=8` by default.

Useful options for `answer_rag.py`:

- `--dry-run` prepares the context without calling Gemini.
- `--offline` uses cached E5/reranker files; generation still needs internet.
- `--strategy fixed` selects the other chunking strategy.
- `--model MODEL_ID` overrides the configured Gemini model.
- `--prompt-version rag_answer_v1_2` replays an earlier prompt.
- `--api-test` makes a minimal Gemini call without E5 or Qdrant.

The answer JSON contains `status`, `answer`, `citations` and `limitations`.
The requested statuses are `answered`, `partial` and `insufficient`. Python
checks source identifiers, exact quote presence, duplicate citations and
agreement between the answer references and citation list. Traces are saved
in `data/answers/`.

Prompt v1.3 adds explicit abstention rules and few-shot examples to distinguish
missing information from an explicitly documented absence. It also asks the
model to ignore instructions inside retrieved documents. These are prompt
instructions, not guarantees of correct behavior.

## Retrieval evaluation

The v2 dataset contains 30 manually annotated questions:

| Split | With expected evidence | Without an answer | Total |
| --- | ---: | ---: | ---: |
| Development | 17 | 3 | 20 |
| Test | 8 | 2 | 10 |

The 14 previously used questions stay in development. Families are distinct
between development and test, but documents and some pages are shared.
The validator checks file versions, evidence references, 36 copied proof
quotes and split consistency.

After the models are cached, run development evaluation:

```bash
python scripts/validate_evaluation.py
python scripts/evaluate_retrieval.py --split dev --compare-reranking \
  --rerank-input metadata --candidate-k 20 --top-k 8 --offline
```

Keep the chosen settings fixed before running the test:

```bash
python scripts/evaluate_retrieval.py --split test --compare-reranking \
  --rerank-input metadata --candidate-k 20 --top-k 8 --offline
```

Remove `--offline` if the models have not been downloaded yet. The evaluator
does not call Gemini. Reports, comparison CSV files and per-question contexts
are saved under `data/evaluation/results/v2/`.

Our final comparison uses 8 passages for every configuration. Dense retrieval
returns 8 directly; reranking retrieves 20 candidates before selecting 8.

| Configuration | DEV evidence coverage | DEV MRR | TEST evidence coverage |
| --- | ---: | ---: | ---: |
| Fixed + dense | 13/17 | 0.760 | 8/8 |
| Fixed + reranking | 16/17 | 0.863 | 8/8 |
| Sections + dense | 14/17 | 0.776 | 8/8 |
| Sections + reranking | 17/17 | 0.953 | 8/8 |

Coverage here requires all annotated evidence groups and their text anchors.
MRR averages `1 / rank` of the first passage from a relevant annotated unit
for each question, with zero if none is found. Questions without an answer are
excluded from these positive retrieval metrics.

We selected sections with reranking from development results. All four
configurations reach 8/8 coverage on the small test set, so the test does not
establish a unique winner. These are results on the initial corpus, not on
every future user PDF.

On the test run, sections with reranking use about 1,760 passage-body tokens
on average, compared with 2,379 for fixed chunks with reranking, around 26% less.
Search plus reranking takes about 444 ms on the tested setup, excluding model
loading, question encoding and generation. These measurements are not total
response times or guarantees for other machines.

## New product PDFs and the API

Private product PDFs are processed separately from the initial corpus.
Ingestion extracts text, splits sections, creates E5 embeddings and indexes
the passages with user, product and document-version metadata.

There is one active PDF per user/product pair. An identical upload with the
same settings can reuse the existing version. A new version becomes active
only after ingestion checks pass; a failed upload keeps the previous ready
version. Older versions remain stored but are excluded from current searches.
The demonstration upload produced 42 chunks from the 9-page Lumera PDF.

Start the API after building the common indexes and caching both models:

```bash
python scripts/configure_rag_api.py
python scripts/run_rag_api.py
```

Configuration adds missing service settings to `.env` without printing or
replacing the existing keys. The server runs one worker at
`http://127.0.0.1:8000`; request schemas are available at `/docs`.

In another terminal with the same environment activated:

```bash
python scripts/call_rag_api.py health
python scripts/call_rag_api.py ingest \
  --user-id demo_user --product-id aqua_lumina \
  --file data/raw/Lumera_Aqua_Lumina_Brand_Product_Book.pdf \
  --title "Aqua Lumina - demonstration" --synthetic
python scripts/call_rag_api.py retrieve \
  --user-id demo_user --product-id aqua_lumina --scope product \
  --question "Quelle est la concentration de niacinamide dans Aqua Lumina ?"
```

Use `answer` instead of `retrieve` to request a Gemini answer. Use
`--document-id` to pin the active version returned by ingestion. CLI identities
are for local tests; application identities must come from the authenticated
backend.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Local service readiness. |
| `POST /products/{product_id}/documents` | Upload a product PDF. |
| `GET /products/{product_id}/documents/current` | Inspect the active document version. |
| `POST /retrieve` | Return ranked passages and provenance, without Gemini. |
| `POST /answer` | Retrieve, generate and validate a sourced JSON answer. |

The scopes are `product`, `common` and `both`. Shared references contain Meta,
ARPP and internal methods; the original fictional Lumera document is excluded
from shared-reference queries. With `both`, private and common results are
merged into a global candidate pool before reranking, without a per-source quota.

Protected routes require a service Bearer token and `X-User-ID` supplied by a
trusted backend. The service token authenticates the caller, not the end user.
Private queries enforce user, product and active-document filters. A stale
document ID is rejected.

## n8n and Docker

The RAG integration separates PDF upload from agent consultation. Upload waits
for ingestion to finish and returns `ready` with the document ID. The backend
keeps that context for the Manager. The Manager can choose a question and an
allowed scope, while identity and document context are fixed outside the LLM's
tool arguments. Product consultation is blocked while documentation is not ready.

n8n uses a separate `X-Workflow-Key` for trusted webhook callers and the API
Bearer credential for calls to Python. Configuration, credentials and endpoint
addresses are still needed after importing workflows. The upload itself does
not launch a campaign.

The Docker setup packages the service, local models and common indexes.
Product documents and answer traces use persistent volumes. The private
demonstration image embeds credentials, which authorized image readers can
extract. A deployment version should be rebuilt without embedded credentials
and receive its secrets at runtime.

We have targeted checks for the RAG API and n8n workflows. They do not establish
an overall success rate for a complete campaign with all specialist agents and
the real interface.

## Tests and known limits

After preparing the corpus and caching the tokenizer, run:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover chunking, ranking behavior, annotation consistency, JSON and
quote validation, product isolation, version changes, API handling and the
local Qdrant lifecycle. Model scores and Gemini calls are simulated in the
relevant unit tests; retrieval quality is measured separately by the evaluator.