# 📄 Document Embedding Analysis

Semantic comparison and feedback signals for long-form scientific and technical document generation.

This project builds structured target documents from papers, patents, and Wikipedia-like sources, then compares generated documents against those targets at multiple levels: outline/plan, section content, references, citation coverage, and general text quality. It is intended both for offline benchmarking and for use as a reward/feedback component during generative optimization.

---

## Table of Contents

1. [Scientific Purpose](#scientific-purpose)
2. [What the Project Measures](#what-the-project-measures)
3. [Installation](#installation)
4. [Environment Configuration](#environment-configuration)
5. [Supported Document Types](#supported-document-types)
6. [Dataset Generation Pipeline](#dataset-generation-pipeline)
7. [Output JSON Schema](#output-json-schema)
8. [How to Generate Target Datasets](#how-to-generate-target-datasets)
9. [How to Compare a Generated Document with a Target](#how-to-compare-a-generated-document-with-a-target)
10. [Using Scores as Feedback for Generative Optimization](#using-scores-as-feedback-for-generative-optimization)
11. [Low-Level DEA Comparison Helpers](#low-level-dea-comparison-helpers)
12. [Configuration Parameters](#configuration-parameters)
13. [Adding New Scientific Documents](#adding-new-scientific-documents)
14. [Adding a New Document Type](#adding-a-new-document-type)
15. [Utility Scripts](#utility-scripts)
16. [Testing](#testing)
17. [Troubleshooting](#troubleshooting)
18. [Recommended Workflows](#recommended-workflows)

---

## Scientific Purpose

Large language models can generate long documents from short prompts, titles, abstracts, or research goals. The central question addressed here is:

> Given only a document title, abstract, topic, or intent, how well can a model reconstruct the document’s structure, section-level content, and bibliography?

The project supports this by converting reference documents into a structured **DEA target JSON**. A generated document can then be compared with the target not only through generic text metrics such as ROUGE or MAUVE, but also through document-aware criteria:

* Does the generated outline resemble the reference outline?
* Are section headings semantically aligned?
* Is the generated content close to the target content?
* Are references and bibliographic resources covered?
* Are citations placed in the right sections?
* Is the generated document useful as a scientific or technical reconstruction?

The intended uses are:

* benchmarking long-document generation systems;
* evaluating semantic structure preservation;
* measuring citation and bibliography coverage;
* producing scalar or multi-objective feedback for optimization loops;
* comparing different prompting, planning, or retrieval strategies.

---

## What the Project Measures

The project distinguishes between **DEA document-aware evaluation** and generic text similarity metrics.

### DEA evaluation dimensions

| Dimension                   | What it evaluates                                           | Typical signal                                    |
| --------------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| Plan / outline              | Similarity between target and generated section structure   | Plan embedding similarity, heading similarity     |
| Sections                    | Alignment of generated headings with target headings        | Section-level embedding similarity                |
| Content                     | Similarity of generated section bodies to reference content | Content embedding similarity                      |
| References / bibliography   | Whether the generated resources match target resources      | Resource embedding similarity, Hungarian matching |
| Citation coverage           | Whether resources are cited and used in relevant sections   | Citation/resource coverage scores                 |
| Length and structure ratios | Whether the generated document has comparable granularity   | Section-count and content-length ratios           |

These dimensions are more informative for long-form generation than a single lexical overlap score. A generated article can have low ROUGE but good semantic structure, or high lexical overlap but poor bibliography coverage.

### Generic text metrics

The project may also compute:

| Metric                              | Role                                                               | Limitation                                                           |
| ----------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------- |
| ROUGE / ROUGE-L                     | Lexical overlap and longest common subsequence similarity          | Sensitive to wording; weak for semantic paraphrase                   |
| MAUVE                               | Distribution-level divergence between generated and reference text | Useful for global text quality, less interpretable per document part |
| Cosine similarity                   | Semantic similarity in embedding space                             | Depends on embedding model and chunking strategy                     |
| Prometheus / WriteHere-style scores | Optional rubric-based article-quality scores                       | Usually requires an LLM or heuristic fallback                        |

The README intentionally treats ROUGE and MAUVE as supplementary metrics. The main value of this repository is the DEA structured comparison: plan, sections, content, references, and citations.

---

## Installation

The project supports a lightweight base install and optional feature groups.

```bash
pip install -e .                    # lightweight base
pip install -e '.[comparison]'       # MAUVE / ROUGE comparison helpers
pip install -e '.[local-embeddings]' # local HuggingFace / Torch embeddings
pip install -e '.[pandoc]'           # Markdown -> LaTeX conversion via pypandoc
pip install -e '.[full]'             # everything
```

For older environments that do not use optional dependencies, the legacy installation path is:

```bash
pip install -r requirements.txt
```

Recommended development setup:

```bash
git clone https://github.com/doxav/document_embedding_analysis.git
cd document_embedding_analysis
python -m venv .venv
source .venv/bin/activate
pip install -e '.[comparison]'
```

Install `.[local-embeddings]` only when you need to generate HuggingFace embeddings locally.

---

## Environment Configuration

Create a `.env` file when using OpenAI embeddings or LLM-based title generation.

```bash
cp .env_template .env
```

Then set:

```bash
OPENAI_API_KEY=sk-your-key-here
```

The project can use:

* OpenAI embeddings, usually `text-embedding-ada-002`;
* HuggingFace embeddings, usually `nomic-ai/nomic-embed-text-v1`;
* GPT-based title generation for splitting long sections, depending on the configured extraction path.

Local HuggingFace embedding generation may require Torch and a CUDA-compatible environment for reasonable speed.

---

## Supported Document Types

The project supports two related benchmark styles:

* **Native DEA / open retrieval:** the model starts from the target title, abstract, topic, and bibliography-style references. Source documents are not bundled; a system must retrieve them or reason from open references.
* **Source-provided summarization:** the model is given source documents or source abstracts and must synthesize the target output.

**Documented local snapshot:** 100 LaTeX `.tex`/`.bib` pairs, 10 arXiv PDFs, 10 configured Wikipedia URLs, 25 patent text files, 0 FreshWiki dumps. Existing generated outputs currently contain 10 LaTeX, 10 arXiv, 12 Wikipedia, and 24 patent DEA JSON files.

| Dataset / Type | Entry point | Input | Target output | Source-document availability | Notes |
| --- | --- | --- | --- | --- | --- |
| BigSurvey `split` archive | `scripts/import_bigsurvey.py --archive split` | Google Drive archive / local archive | Survey sections | **Provided** as cited-paper abstracts | Real Drive import tested; groups by `paper_id`; sample test split items had 17-112 source abstracts. |
| BigSurvey `original` archive | `scripts/import_bigsurvey.py --archive original` | Google Drive archive / local archive | Survey sections | **Provided** as cited-paper abstracts | Real Drive import tested; archive lacks `paper_id`, so importer groups by title. |
| MultiLexSum | `scripts/import_multilexsum.py` | Official release download or local JSON files | Long legal summary | **Provided** as legal case documents | Default mode fetches official `sources.json` plus `{train,dev,test}.json` split files; local `--records-json`/`--sources-json` mode is available for offline runs. |
| LaTeX | `main.py` option `l` | `.tex` + `.bib` | Paper reconstruction DEA | **Partial retrieval locators only** | BibTeX may contain DOI/URL fields; these are open-retrieval locators, not bundled source documents. |
| arXiv PDF | `main.py` option `a` | PDF files | Paper reconstruction DEA | **Not provided** | Parsed references are bibliographic text only; use open retrieval. |
| Wikipedia | `main.py` option `w` | Wikipedia URLs | Article reconstruction DEA | **Not document-provided** | New extraction preserves external citation URLs as retrieval hints where Wikipedia exposes them, but coverage is incomplete and should be treated as open retrieval. |
| Patents | `main.py` option `p` | EPO-style patent text | Patent reconstruction DEA | **Not provided** | Extracted references are patent-number strings only; use open retrieval. |
| FreshWiki | `main.py` option `f` | FreshWiki JSON dumps | Article reconstruction DEA | Unknown locally | No local FreshWiki data is present in this checkout. |
| Enriched DEA second pass | `scripts/enrich_dea_sources.py` | Existing DEA JSON | File-backed promptfoo rows | Depends on resource fields | Works only when resources contain real local paths, source URLs, or DOI/URL locators; otherwise rows are marked `reference_only_open_retrieval`. |

---

## Dataset Generation Pipeline

The extraction pipeline converts heterogeneous source documents into a common DEA JSON format.

1. **Extraction**
   Type-specific parsers extract title, abstract/context, headings, section content, and references.

2. **Chunking**
   Sections longer than `MAX_EMBEDDING_TOKEN_LENGTH` are split into smaller chunks.

3. **Heading repair / title generation**
   Split sections receive generated or suffixed headings. Reference sections keep the original heading with numeric suffixes.

4. **Reference tracking**
   Citations are mapped to `resource_id` values where possible.

5. **Dual embedding generation**
   The system can generate both OpenAI and HuggingFace embeddings for headings, section content, and resources.

6. **JSON export**
   Structured target files are written to `output/{type}/`.

---

## Output JSON Schema

Each generated target is a DEA JSON object. The exact fields may vary by document type, but the expected structure is:

```json
{
  "id": "uuid-or-source-id",
  "title": "Document title",
  "abstract": "Document abstract or context",
  "plan": [
    {
      "section_id": 1,
      "section": "Introduction",
      "content": "Section text...",
      "section_embedding_1": [0.1, 0.2],
      "section_embedding_2": [0.1, 0.2],
      "content_embedding_1": [0.1, 0.2],
      "content_embedding_2": [0.1, 0.2],
      "resources_used": [1, 3]
    }
  ],
  "resources": [
    {
      "resource_id": 1,
      "resource_description": "Reference or citation text",
      "resource_embedding_1": [0.1, 0.2],
      "resource_embedding_2": [0.1, 0.2]
    }
  ],
  "plan_embedding_1": [0.1, 0.2],
  "plan_embedding_2": [0.1, 0.2],
  "embedding1_model": "text-embedding-ada-002",
  "embedding2_model": "nomic-embed-text-v1",
  "success": true
}
```

The two embedding families are conventionally named:

* `_embedding_1`: OpenAI embedding;
* `_embedding_2`: HuggingFace embedding.

---

## How to Generate Target Datasets

Run the interactive extraction script:

```bash
python main.py
```

Prompt options:

```text
Enter  -> process all document types
1      -> process one example of each type
l      -> process LaTeX documents
a      -> process arXiv PDFs
w      -> process Wikipedia URLs
p      -> process patent files
f      -> process FreshWiki JSON dumps
```

Example:

```text
Generate for all (Enter), 1 for each type (1), or just an example of a given type
(Latex: L, Arxiv: A, Wikipedia: W, Patent: P, FreshWiki: F): w

[INFO] Title: Large language model
[INFO] Abstract: A large language model (LLM) is a type of...
[INFO] 1/15 - created section + content embeddings for h2 Definitions
[INFO] Successfully extracted plan and content
[INFO] Written to file: output/wikipedia/Large language model.json
```

Outputs are written to:

```text
output/
├── latex/
├── arxiv/
├── wikipedia/
├── freshwiki/
└── patent/
```

### Generating a target for optimization

A typical feedback loop needs:

1. a target DEA JSON generated from a reference document;
2. a candidate document generated by an LLM or agent;
3. an evaluator call that returns structured similarity scores;
4. a scalar or multi-objective score used by the optimizer.

For example, the target can be generated once from a Wikipedia article, paper, or patent, then reused across many candidate generations.

---

## How to Compare a Generated Document with a Target

Use `evaluate_document` when your candidate is raw Markdown or LaTeX and your target is a DEA JSON object.

```python
import json
from common.doc_eval import evaluate_document

with open("output/wikipedia/Large language model.json", encoding="utf-8") as f:
    solution = json.load(f)

candidate_markdown = """
# Large language model

## Introduction
A large language model is a neural language model trained on large text corpora.

## Architecture
Most modern LLMs use transformer architectures.

## Applications
They are used for question answering, summarization, coding, and writing.
"""

scores = evaluate_document(
    document_content=candidate_markdown,
    solution=solution,
    content_type="markdown",
    skip_dea=False,
    use_enhanced_metrics=False,
    dea_embedding_backend="hf",   # or "openai"
)

print(scores["dea_evaluation_scores"])
print(scores["article_metrics"])
```

Use `content_type="latex"` for a LaTeX candidate:

```python
scores = evaluate_document(
    document_content=latex_source,
    solution=solution,
    content_type="latex",
    skip_dea=False,
)
```

### Key `evaluate_document` parameters

| Parameter               | Meaning                                                 |
| ----------------------- | ------------------------------------------------------- |
| `document_content`      | Candidate document text                                 |
| `solution`              | DEA target JSON dictionary                              |
| `content_type`          | Usually `"markdown"` or `"latex"`                       |
| `skip_dea`              | If `True`, skip target-distance evaluation              |
| `use_enhanced_metrics`  | If `True`, attempt Prometheus / WriteHere-style scoring |
| `openai_model`          | Optional model for LLM-based enhanced scoring           |
| `dea_embedding_backend` | `"hf"`, `"openai"`, or auto-detect                      |
| `dea_embedding_model`   | Optional override for the embedding model               |
| `skip_entity_recall`    | Avoid entity recall extraction when not needed          |
| `golden_entities`       | Optional precomputed target entities                    |
| `use_dea_judge`         | If `True`, run the optional DEA-aware qualitative judge when a model/client/lm is available |
| `dea_judge_model`       | Optional LLM name for the judge, for example `gpt-5-nano` or an OpenRouter model id |
| `dea_judge_client`      | Optional OpenAI-compatible client for the judge, including OpenRouter clients |
| `dea_judge_lm`          | Optional callable fake/local judge used instead of an API client |

`evaluate_document` is the recommended high-level API for feedback during generation because it can return DEA target-distance scores and general article metrics from a single call.

### Optional DEA-aware qualitative judge

`evaluate_document(...)` can also return a compact `dea_judge` object. The judge is comparative: it receives DEA scores, article metrics, gold plan/bibliography excerpts, candidate plan/bibliography excerpts, and selected weak-looking candidate sections, then returns only observations for the optimizer to interpret.

```python
scores = evaluate_document(
    document_content=candidate_markdown,
    solution=solution,
    content_type="markdown",
    skip_dea=True,              # keep demos fast; set False for full DEA scoring
    use_dea_judge=True,
    dea_judge_model="gpt-5-nano",
)

print(scores["dea_judge"]["status"])
print(scores["dea_judge"]["qualitative_assessment"])
print(scores["dea_judge"]["problems"])
```

If no judge model, client, or callable is provided, the field is returned with `status="skipped"`. Invalid JSON or LLM errors return `status="error"` without interrupting the rest of the evaluation. A step-by-step notebook demo is available at `scripts/step_by_step_demo.ipynb`.

---

## Using Scores as Feedback for Generative Optimization

DEA scores can be used directly as reward signals or optimization feedback.

A common pattern is to compute all available metrics, then define a task-specific scalar objective.

```python
def score_candidate(candidate_markdown: str, solution: dict) -> float:
    scores = evaluate_document(
        document_content=candidate_markdown,
        solution=solution,
        content_type="markdown",
        skip_dea=False,
        use_enhanced_metrics=False,
        dea_embedding_backend="hf",
        skip_entity_recall=True,
    )

    dea = scores.get("dea_evaluation_scores", {})
    article = scores.get("article_metrics", {})

    plan = (
        dea.get("plan_embedding_similarity")
        or dea.get("section_total_similarity")
        or dea.get("global_plan_embedding_similarity")
        or 0.0
    )
    content = (
        dea.get("plan_contents_embedding_similarity")
        or dea.get("content_total_similarity")
        or dea.get("global_plan_contents_embedding_similarity")
        or 0.0
    )
    references = (
        dea.get("plan_resources_embedding_similarity")
        or dea.get("resources_citation_coverage_score")
        or dea.get("bibliography_coverage1")
        or 0.0
    )
    rouge_l = (
        article.get("rouge_scores", {})
        .get("rouge-l", {})
        .get("f", 0.0)
    )

    return (
        0.30 * plan
        + 0.45 * content
        + 0.15 * references
        + 0.10 * rouge_l
    )
```

Suggested optimization objectives:

| Objective                         | Suggested weighting                                                     |
| --------------------------------- | ----------------------------------------------------------------------- |
| Outline reconstruction            | High plan / section-heading weight                                      |
| Scientific article reconstruction | High content + reference weight                                         |
| Citation-aware generation         | High bibliography and citation-coverage weight                          |
| Retrieval-augmented generation    | High reference coverage + content similarity                            |
| General writing quality           | ROUGE / entity recall / enhanced rubric scores as supplementary metrics |

For multi-objective optimization, keep the dimensions separate instead of reducing them to a scalar:

```python
def score_candidate_multiobjective(candidate_markdown: str, solution: dict) -> dict:
    scores = evaluate_document(
        candidate_markdown,
        solution=solution,
        content_type="markdown",
        skip_dea=False,
        use_enhanced_metrics=False,
    )
    dea = scores.get("dea_evaluation_scores", {})
    article = scores.get("article_metrics", {})
    return {
        "dea": dea,
        "article": article,
    }
```

---

## Low-Level DEA Comparison Helpers

Use low-level helpers when both the target and candidate are already DEA JSON documents with embeddings.

```python
import json
from common.test_utils import compare_documents_sections, compare_documents_content

with open("output/wikipedia/target.json", encoding="utf-8") as f:
    target = json.load(f)

with open("output/wikipedia/candidate.json", encoding="utf-8") as f:
    candidate = json.load(f)

plan_scores = compare_documents_sections(target, candidate)
content_scores = compare_documents_content(target, candidate)

print(plan_scores)
print(content_scores)
```

These functions are lower-level than `evaluate_document`:

* `compare_documents_sections(...)` compares section headings / plan structure;
* `compare_documents_content(...)` compares section body content;
* both include embedding similarities and bibliography coverage where available.

Use these helpers for debugging DEA JSON extraction or comparing two already-embedded document representations.

---

## Configuration Parameters

Important settings live in `common/config.py`.

```python
MAX_EMBEDDING_TOKEN_LENGTH = 512
ALLOW_parallel_gen_embed_section_content = True
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
HUGGINGFACE_EMBEDDING_PATH = "nomic-ai/nomic-embed-text-v1"
HUGGINGFACE_EMBEDDING_MODEL_NAME = "nomic-embed-text-v1"
```

### Processing behavior

* Documents must contain enough sections after extraction and chunking to be meaningful.
* Sections above the embedding token limit are split recursively.
* Reference sections are split with numeric suffixes.
* Non-reference split sections may receive generated concise headings.
* Parallel embedding generation can speed up extraction but may require memory tuning.

---

## Adding New Scientific Documents

### LaTeX papers

Place `.tex` and `.bib` pairs in `data/latex/`:

```bash
cp your_paper.tex data/latex/
cp your_paper.bib data/latex/
python main.py   # select l
```

Or use the downloader:

```bash
python scripts/select_arxiv_latex.py --max-papers 10
```

### arXiv PDFs

Place PDFs in `data/arxiv/`:

```bash
cp paper.pdf data/arxiv/
python main.py   # select a
```

### Wikipedia articles

Add one URL per line to `data/wiki_urls.txt`:

```bash
echo "https://en.wikipedia.org/wiki/Your_Topic" >> data/wiki_urls.txt
python main.py   # select w
```

### Patents

Place tab-delimited patent text files in `data/patents/`, or split an EPO extract:

```bash
python scripts/split_epo_extract.py data/extract.txt data/patents
```

Expected patent sections include:

```text
TITLE
DESCR
CLAIM
```

---

## Adding a New Document Type

To add a new format such as PubMed XML or journal HTML:

1. Add a new enum entry in `common/utils.py`.
2. Create a new processor in `common/doc_newtype.py`.
3. Implement `extract_paper_data()`.
4. Register the new type in `main.py`.
5. Add tests with a small fixture.

Example skeleton:

```python
from common.doc_base import Document
from common.utils import DocType


class DocPubmed(Document):
    def __init__(self, source: str, logger, output_dir=None):
        super().__init__(source, DocType.PUBMED, logger, output_dir)

    def extract_paper_data(self):
        paper_dict = {
            "title": "...",
            "abstract": "...",
            "Introduction": "...",
            "Methods": "...",
            "Results": "...",
        }
        refs = [
            {"resource_id": 1, "resource_description": "..."},
        ]
        self.setReferences(refs)
        return paper_dict

    def extract_plan_and_content(self, skip_if_exists=False):
        super().generateOutputFile()
        paper_data = self.extract_paper_data()
        paper_data = self.divide_into_chunks(paper_data)

        if len(paper_data.keys()) < 3:
            self.getLogger().error("Document too small; need at least 3 sections")
            return None

        plan_json = self.generate_embeddings_plan_and_section_content(paper_data)
        self.writeOuputJson(plan_json)
        return plan_json
```

Document extraction guidelines:

* Preserve hierarchical headings where possible.
* Normalize references to sequential `resource_id` values.
* Track citations into `resources_used` for each section.
* Provide fallbacks for missing abstracts or references.
* Remove markup while preserving scientific content and section boundaries.

---

## Utility Scripts

### ArXiv LaTeX downloader

```bash
python scripts/select_arxiv_latex.py
python scripts/select_arxiv_latex.py --max-papers 50 --out-dir data/latex
python scripts/select_arxiv_latex.py --max-papers 20 --dry-run
```

Features:

* queries arXiv for relevant papers;
* filters for surveys or major-conference-style papers;
* downloads LaTeX source tarballs;
* keeps papers with exactly one `.tex` and one `.bib` file;
* writes files using a title and arXiv ID naming pattern.

### EPO patent splitter

```bash
python scripts/split_epo_extract.py <input_extract.txt> <output_directory>
```

Example:

```bash
python scripts/split_epo_extract.py \
    data/2022week30_EP0600000_extract.txt \
    data/patents
```

The splitter:

* extracts individual English-language patent records;
* keeps documents with required `TITLE`, `DESCR`, and `CLAIM` sections;
* sanitizes filenames;
* skips existing files.

---

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Run collection only:

```bash
python -m pytest --collect-only -q
```

Run legacy comparison script if you have generated JSON outputs:

```bash
python tests.py
```

The pytest suite should be able to import the package without requiring heavyweight optional dependencies unless tests explicitly exercise those features.

---

## Troubleshooting

### Importing `common.doc_eval` requires a missing package

The package is designed so heavyweight dependencies should be loaded lazily. If importing `common.doc_eval` requires Torch, Flair, pypandoc, or sentence-transformers, check for accidental top-level imports.

### `pypandoc` is missing

Install Pandoc support only when Markdown-to-LaTeX conversion is needed:

```bash
pip install -e '.[pandoc]'
```

### Local HuggingFace embeddings are missing dependencies

Install:

```bash
pip install -e '.[local-embeddings]'
```

### MAUVE or ROUGE comparison helpers are missing

Install:

```bash
pip install -e '.[comparison]'
```

### Pytest collects archival files

If an `old/` directory contains historical scripts, exclude it in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
norecursedirs = ["old"]
```

---

## Recommended Workflows

### Workflow A: Build a target dataset

```bash
pip install -e '.[local-embeddings]'
python main.py
```

Use this when you want to create DEA JSON targets from papers, PDFs, patents, or Wikipedia articles.

### Workflow B: Evaluate generated documents

```bash
pip install -e '.[comparison]'
```

Then call `evaluate_document(...)` from Python to compare Markdown or LaTeX candidates to a DEA target.

### Workflow C: Use DEA as an optimization reward

1. Generate or load a DEA target JSON.
2. Generate a candidate document from your model or agent.
3. Call `evaluate_document(...)`.
4. Extract plan/content/reference metrics.
5. Feed the resulting scalar or vector back into your optimizer.

This is the preferred path for training-time or search-time feedback because it preserves interpretable sub-scores rather than reducing the document comparison to a single generic overlap metric.

## Source-Provided MDS/QFS and RAS-Stage Benchmarks

This repo now supports both native DEA reconstruction targets and source-provided summarization-stage tasks.

- **Open-search/native DEA**: the model starts from title/abstract/topic intent and retrieves/searches to reconstruct the gold target.
- **Source-provided**: the model is given source documents and must synthesize the target long-form output (MDS/QFS/RAS stage).

### Source Availability Matrix

| Dataset | Source-provided benchmark? | Open retrieval needed? | Current behavior |
| --- | --- | --- | --- |
| BigSurvey `split` | Yes | No, for cited abstracts | Downloads from Drive alias `split`; writes promptfoo rows with source abstract files. |
| BigSurvey `original` | Yes | No, for cited abstracts | Downloads from Drive alias `original`; writes promptfoo rows with source abstract files, grouped by title. |
| MultiLexSum | Yes | No | Official mode downloads `sources.json` and `{split}.json`; local mode reads `--records-json` and `--sources-json`. |
| LaTeX DEA outputs | No, not as bundled documents | Yes | BibTeX DOI/URL fields become best-effort retrieval locators; manifests use `source_locator_best_effort` when locators exist. |
| arXiv DEA outputs | No | Yes | Existing parsed PDF references contain citation text only; manifests use `reference_only_open_retrieval` or `no_resources`. |
| Wikipedia DEA outputs | No | Yes | External citation URLs are retrieval hints only; older checked-in outputs are citation-text only. |
| Patent DEA outputs | No | Yes | References are extracted patent identifiers only; manifests use `reference_only_open_retrieval` or `no_resources`. |
| FreshWiki | Unknown locally | Unknown | No local dumps are present. |
| Generic enriched DEA | Conditional | Conditional | Only source-backed when resources contain local paths, HTTP URLs, or DOI/URL locators. |

### Example CLIs
- `python scripts/import_bigsurvey.py --archive split --n 10 --split test --output-dir benchmark/source_provided/bigsurvey_split --skip-embeddings`
- `python scripts/import_bigsurvey.py --archive original --n 10 --split test --output-dir benchmark/source_provided/bigsurvey_original --skip-embeddings`
- `python scripts/import_multilexsum.py --n 20 --split test --output-dir benchmark/source_provided/multilexsum --skip-embeddings`
- `python scripts/import_multilexsum.py --source local --records-json data/multilexsum/test.json --sources-json data/multilexsum/sources.json --n 20 --output-dir benchmark/source_provided/multilexsum_local --skip-embeddings`
- `python scripts/enrich_dea_sources.py --dea-root output/latex --output-dir benchmark/open_retrieval/latex --fetch-remote --continue-on-error`
- `python scripts/enrich_dea_sources.py --dea-root output/arxiv --output-dir benchmark/open_retrieval/arxiv --continue-on-error`

### Source resolver behavior and limits
- Resolver flags: `--fetch-remote`, `--copy-local`, `--no-copy-local`, `--continue-on-error`, `--overwrite`.
- Importer flag: `--skip-embeddings` is accepted for compatibility with dataset-building scripts.
- Source mode is recorded in `dataset_manifest.json`, item `metadata.json`, and `dea_solution.json` as `source_document_mode`.
- `reference_only_open_retrieval` means the dataset has citation text but no source document path or URL.
- `source_locator_best_effort` means DOI/URL/path fields exist, but they are retrieval locators rather than guaranteed bundled source documents.
- `no_resources` means no bibliography/resource entries were extracted.
- BigSurvey sources are cited-paper abstracts (not full PDFs).
- Multi-LexSum sources are dataset-exposed legal source texts. Official mode caches the large upstream `sources.json` under `.cache/multilexsum` and stream-selects only needed documents for output.

### Adding a new dataset
1. Create `TaskBundle` records.
2. Write with `write_dataset(...)` for canonical layout + `promptfoo_dataset.jsonl`.
3. Optionally convert/export DEA-compatible `dea_solution.json`.


### Generation-time enrichment status
- Generation-time wrapper is currently disabled because `main.py` is interactive-only.
- Use second-pass enrichment via `scripts/enrich_dea_sources.py`.
