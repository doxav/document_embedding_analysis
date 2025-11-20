# 📄 Document Embedding Analysis
*Semantic comparison of scientific documents using dual embedding models and advanced text similarity metrics.*

## 🎯 Scientific Purpose

This project addresses a challenge in evaluating large language models: **Given only a document's title and abstract, how well can an LLM reconstruct the document's structure, content and bibliography?**

The system extracts structured content from diverse scientific and technical documents, generates dual embeddings (OpenAI + HuggingFace), and performs comprehensive semantic comparison using:

- **ROUGE-L**: Measures lexical overlap and longest common subsequence similarity
- **MAUVE**: Quantifies text distribution divergence between generated and reference content
- **Cosine Similarity**: Computes semantic similarity in embedding space
- **Hungarian Algorithm**: Optimally matches bibliographic references across documents

This dataset enables researchers to benchmark LLM content generation capabilities, evaluate semantic structure preservation, and analyze citation coverage in generated scientific text.

## 📚 Supported Document Types

The system processes five scientific document formats through specialized extractors:

**Current Dataset:** 99 LaTeX papers, 10 arXiv PDFs, 10 Wikipedia URLs, 25 Patents, 0 FreshWiki dumps

| Type | Identifier | Description | Input Format | Key Features |
|------|------------|-------------|--------------|--------------|
| **LaTeX** | `l` | Academic papers with BibTeX | `.tex` + `.bib` files | Hierarchical section numbering (1, 1.1, 1.1.1), `\cite{}` extraction |
| **arXiv** | `a` | PDF research papers | PDF files | Regex-based abstract/reference extraction, citation pattern `[N]` |
| **Wikipedia** | `w` | Wiki articles (live) | URLs | BeautifulSoup scraping, hierarchical headings (`h2 Title - Subtitle`) |
| **FreshWiki** | `f` | Wiki articles (JSON dumps) | `.json` files | Pre-formatted Wikipedia content from dumps |
| **Patents** | `p` | Patent documents | Tab-delimited `.txt` | Multi-claim parsing, international patent number extraction (US/EP/WO/JP/CN/DE) |

### Document Processing Pipeline

1. **Extraction**: Type-specific parser extracts title, abstract, sections, content, and references
2. **Chunking**: Sections >512 tokens are split using `RecursiveCharacterTextSplitter`
3. **Title Generation**: GPT-4o-mini auto-generates 7-word max titles for split sections (except references)
4. **Dual Embedding**: Both OpenAI (`text-embedding-ada-002`) and HuggingFace (`nomic-embed-text-v1`) models generate embeddings
5. **Reference Tracking**: Citations mapped to bibliography with resource IDs and embeddings
6. **JSON Export**: Structured output with plan/content embeddings saved to `output/{type}/`

## 💻 How to Run the Code

### 1. Installation

```bash
git clone https://github.com/doxav/document_embedding_analysis.git
cd document_embedding_analysis
pip install -r requirements.txt
```

### 2. Environment Configuration

**Required**: OpenAI API key for embeddings and title generation

```bash
# Copy template
cp .env_template .env

# Edit .env and add your key:
OPENAI_API_KEY=sk-your-key-here
```

The system uses:
- **OpenAI API**: `text-embedding-ada-002` embeddings + GPT-4o-mini for title generation
- **HuggingFace**: `nomic-ai/nomic-embed-text-v1` (local inference, auto-detects CUDA)

### 3. Running the Extraction

```bash
python main.py
```

**Interactive Prompt Options**:
- `Enter` (empty): Process all document types (LaTeX, arXiv, Wikipedia, Patents, FreshWiki)
- `1`: Process one example of each type, then stop
- `l`: Process only LaTeX documents
- `a`: Process only arXiv PDFs
- `w`: Process only Wikipedia URLs
- `p`: Process only Patent files
- `f`: Process only FreshWiki JSON dumps

**Example Session**:
```
Generate for all (Enter), 1 for each type (1), or just an example of a given type 
(Latex: L, Arxiv: A, Wikipedia: W, Patent: P, FreshWiki: F): w

[INFO] Title: Large language model
[INFO] Abstract: A large language model (LLM) is a type of...
[INFO] 1/15 - created section + content embeddings for h2 Definitions
[INFO] Successfully extracted plan and content
[INFO] Written to file: output/wikipedia/Large language model.json
[INFO] Time taken: 2mins 34.5s
```

### 4. Running Comparison Tests

```bash
python tests.py
```

Tests compare identical and different documents using all similarity metrics. Requires existing JSON outputs in `output/` directories.

## ⚙️ Configuration Parameters

### Embedding Settings (`common/config.py`)

```python
# Token limit for single embedding (hard constraint)
MAX_EMBEDDING_TOKEN_LENGTH = 512

# Enable parallel batch embedding (recommended: True)
# Falls back to sequential if HuggingFace OOM occurs
ALLOW_parallel_gen_embed_section_content = True

# Embedding models (change if needed)
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
HUGGINGFACE_EMBEDDING_PATH = "nomic-ai/nomic-embed-text-v1"
```

### Processing Behavior

- **Minimum Sections**: Documents must have ≥3 sections after chunking (validation enforced)
- **Token Counting**: Uses `tiktoken` with `gpt-4o-mini` encoding
- **Parallel Batching**: Automatically splits batches exponentially (n *= 2) on OOM errors
- **Reference Sections**: Split sections append ` 1`, ` 2` suffixes (preserves original heading)
- **Other Sections**: Use GPT-4o-mini to generate concise titles (max 7 words)

### Output Structure

```
output/
├── latex/          # .tex processed files
├── arxiv/          # PDF processed files  
├── wikipedia/      # Live Wikipedia scrapes
├── freshwiki/      # JSON Wikipedia dumps
└── patent/         # Patent document outputs
```

Each JSON contains: `id`, `title`, `abstract`, `plan[]`, `resources[]`, `plan_embedding_1/2`, embedding model metadata, and `success` status.

## ➕ Adding New Scientific Documents

### Quick Add to Existing Types

**LaTeX Papers**: Place `.tex` + `.bib` pairs in `data/latex/` (manually) or use automated downloader:
```bash
# Manual addition
cp your_paper.tex data/latex/
cp your_paper.bib data/latex/
python main.py  # Select 'l'

# Or download from arXiv automatically
python scripts/select_arxiv_latex.py --max-papers 10
```

**arXiv PDFs**: Place PDFs in `data/arxiv/`
```bash
cp paper.pdf data/arxiv/
python main.py  # Select 'a'
```

**Wikipedia Articles**: Edit `data/wiki_urls.txt` (one URL per line)
```bash
echo "https://en.wikipedia.org/wiki/Your_Topic" >> data/wiki_urls.txt
python main.py  # Select 'w'
```

**Patents**: Split EPO extract files or place tab-delimited `.txt` files in `data/patents/`
```bash
# Split EPO extract into individual patents
python scripts/split_epo_extract.py data/extract.txt data/patents

# Format: \tLANG\tSECTION_TYPE\tCONTENT
# Required sections: TITLE, DESCR, CLAIM
```

### Creating a New Document Type

To add support for a new scientific format (e.g., PubMed XML, journal articles):

1. **Define Document Type** in `common/utils.py`:
```python
class DocType(Enum):
    # ... existing types
    PUBMED = "pubmed"
```

2. **Create Document Processor** as `common/doc_pubmed.py`:
```python
from common.doc_base import Document
from common.utils import DocType

class DocPubmed(Document):
    def __init__(self, source: str, logger, output_dir=None):
        super().__init__(source, DocType.PUBMED, logger, output_dir)
    
    def extract_paper_data(self):
        # Return dict with keys: title, abstract, sections (dict), references (list)
        # sections dict format: {"Section Heading": "content text", ...}
        # references list format: [{"resource_id": 1, "resource_description": "..."}, ...]
        
        paper_dict = {
            "title": "...",
            "abstract": "...",
            "Section 1": "content...",
            # ... more sections
        }
        
        refs = [{"resource_id": i, "resource_description": "..."} for i in ...]
        self.setReferences(refs)
        
        return paper_dict
    
    def extract_plan_and_content(self, skip_if_exists=False):
        super().generateOutputFile()
        paper_data = self.extract_paper_data()
        paper_data = self.divide_into_chunks(paper_data)  # Auto-handles >512 token sections
        
        # Validate minimum sections
        if len(paper_data.keys()) < 3:
            self.getLogger().error("Document too small (need ≥3 sections)")
            return
        
        plan_json = self.generate_embeddings_plan_and_section_content(paper_data)
        self.writeOuputJson(plan_json)
        return plan_json
```

3. **Register in `main.py`**:
```python
from common.doc_pubmed import DocPubmed

def process_files(file_type: str, files: List, output_preference: str):
    # ... existing if statements
    elif file_type == "m":  # PubMed
        doc = DocPubmed(file, logging)

# In main():
processing_map = {
    # ... existing mappings
    "m": Path("data/pubmed").glob("*.xml"),
}

types_to_process = (
    ["l", "a", "w", "p", "f", "m"]  # Add new type
    if output_preference in ["", "1"]
    else [output_preference]
)
```

4. **Test Your Processor**:
```bash
mkdir -p data/pubmed
cp sample.xml data/pubmed/
python main.py  # Select 'm'
```

### Document Extraction Guidelines

- **Extract hierarchical structure**: Preserve heading levels where applicable
- **Normalize references**: Use sequential `resource_id` (1, 2, 3...)
- **Track citations**: Populate `resources_used` in sections with cited reference IDs
- **Handle missing data**: Provide defaults (`"no abstract"` if abstract missing)
- **Clean content**: Remove markup/formatting but preserve text structure

## 🛠️ Utility Scripts

### ArXiv LaTeX Downloader (`scripts/select_arxiv_latex.py`)

Automatically download high-quality arXiv papers with LaTeX source code.

**Features:**
- Queries arXiv API for recent cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, stat.ML papers
- Filters for surveys (title/abstract keywords) OR top-conference papers (NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ECCV, ICCV, KDD, AAAI, WWW, SIGIR)
- Downloads LaTeX source tarballs from `https://arxiv.org/e-print/<id>`
- Validates: keeps only papers with exactly **1 .tex + 1 .bib** file
- Conservative filtering (~29% acceptance rate ensures quality)

**Usage:**
```bash
# Download 5 papers (default)
python scripts/select_arxiv_latex.py

# Download 50 papers to custom directory
python scripts/select_arxiv_latex.py --max-papers 50 --out-dir data/latex

# Preview candidates without downloading
python scripts/select_arxiv_latex.py --max-papers 20 --dry-run
```

**Parameters:**
- `--out-dir DIR`: Output directory (default: `data/latex`)
- `--max-papers N`: Maximum papers to download (default: `5`)
- `--dry-run`: Preview candidates without downloading

**Output Format:** `<Title>_arxiv_<ID>.tex` and `<Title>_arxiv_<ID>.bib`
- Example: `CODE-II_A_large-scale_dataset_for_artificial_intelligence_in_ECG_analysis_arxiv_2511.15632v1.tex`

### EPO Patent Splitter (`scripts/split_epo_extract.py`)

Split multi-patent EPO weekly extract files into individual patent documents.

**Usage:**
```bash
python scripts/split_epo_extract.py <input_extract.txt> <output_directory>

# Example
python scripts/split_epo_extract.py \
    data/2022week30_EP0600000_extract.txt \
    data/patents
```

**Input Format:** EPO tab-delimited extract with columns:
- Country, Document number, Kind code, Publication date, Language, Section type, Section number, Content
- Section types: `TITLE`, `DESCR` (description), `CLAIM`, `PDFEP`

**Output:** Individual `.txt` files named `<Title>_<PatentID>.txt`
- Example: `APPARATUS_FOR_THE_MEASUREMENT_OF_ATRIAL_PRESSURE_EP0615422B1.txt`
- Contains only English language sections
- Filters for patents with required sections: `TITLE`, `DESCR`, `CLAIM`
- Skips existing files (no overwrite)

**Notes:**
- See `scripts/README.md` for detailed documentation
- Compatible with `DocPatent` without modifications
- Title sanitization: LaTeX commands removed, special chars → underscores, 80 char max
