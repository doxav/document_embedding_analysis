from dotenv import load_dotenv, find_dotenv


OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
HUGGINGFACE_EMBEDDING_PATH = "nomic-ai/nomic-embed-text-v1"
HUGGINGFACE_EMBEDDING_MODEL_NAME = "nomic-embed-text-v1"
HUGGINGFACE_EMBEDDING_PREFIX = ""
MAX_EMBEDDING_TOKEN_LENGTH = 512
ALLOW_parallel_gen_embed_section_content=True

_ = load_dotenv(find_dotenv())


class _LazyEmbedding:
    """Proxy that initializes the real embedding backend on first use."""

    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def embed_documents(self, *args, **kwargs):
        return self._get().embed_documents(*args, **kwargs)

    def embed_query(self, *args, **kwargs):
        return self._get().embed_query(*args, **kwargs)


def get_openai_embedder():
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI embeddings require `langchain-community`. "
            "Install it only when generating OpenAI embeddings."
        ) from exc

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME)


def get_hf_embedder():
    try:
        import torch
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace embeddings require `torch` and `langchain-community`. "
            "Install them only when generating HF embeddings."
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_PATH,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


embed_OPENAI = _LazyEmbedding(get_openai_embedder)
embed_HF = _LazyEmbedding(get_hf_embedder)
