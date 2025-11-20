import torch
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
HUGGINGFACE_EMBEDDING_PATH = "nomic-ai/nomic-embed-text-v1"
HUGGINGFACE_EMBEDDING_MODEL_NAME = "nomic-embed-text-v1"
HUGGINGFACE_EMBEDDING_PREFIX = ""
MAX_EMBEDDING_TOKEN_LENGTH = 512
ALLOW_parallel_gen_embed_section_content=True

_ = load_dotenv(find_dotenv())
embed_OPENAI = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, )
embed_HF = HuggingFaceEmbeddings(
    model_name=HUGGINGFACE_EMBEDDING_PATH,
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu", "trust_remote_code": True
    },
    encode_kwargs={"normalize_embeddings": True}
)