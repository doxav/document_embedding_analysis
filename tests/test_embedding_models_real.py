import os

import pytest

from common.config import HUGGINGFACE_EMBEDDING_PATH
from common.env.IR_CPS_TechSynthesis.env import DocumentStructure


HF_MODELS_TO_SWITCH = [
    "nomic-ai/nomic-embed-text-v1",
    "jinaai/jina-embeddings-v5-text-nano",
    "nomic-ai/nomic-embed-text-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def _offline_hf():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _local_embedder_or_skip(model_name: str):
    pytest.importorskip("sentence_transformers")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"trust_remote_code": True, "local_files_only": True},
        )
    except Exception as exc:
        pytest.skip(f"{model_name!r} is not available locally: {exc}")


@pytest.mark.parametrize("model_name", HF_MODELS_TO_SWITCH)
def test_hf_embedding_model_exists_locally_and_embeds(model_name):
    _offline_hf()
    embedder = _local_embedder_or_skip(model_name)

    first = embedder.embed_query("embedding routing smoke test")
    second = embedder.embed_query("a different sentence for the same model")

    assert isinstance(first, list)
    assert isinstance(second, list)
    assert len(first) > 0
    assert len(first) == len(second)
    assert all(isinstance(x, float) for x in first[:10])


def test_document_structure_switches_between_real_hf_models_when_available():
    _offline_hf()
    for model_name in HF_MODELS_TO_SWITCH:
        _local_embedder_or_skip(model_name)

    DocumentStructure.embedding_model_cls = None
    DocumentStructure.embedding_model_cls_name = None

    seen_dimensions = {}
    for model_name in HF_MODELS_TO_SWITCH:
        document = DocumentStructure(
            synthesis_type="test",
            initial_goal="test",
            embedding_model_name=model_name,
        )
        seen_dimensions[model_name] = document.embedding_size
        assert document.embedding_model_name == model_name
        assert DocumentStructure.embedding_model_cls_name == model_name
        assert document.embedding_size > 0

    assert set(seen_dimensions) == set(HF_MODELS_TO_SWITCH)


def test_document_structure_legacy_nomic_short_name_maps_exactly_not_by_substring():
    _offline_hf()
    _local_embedder_or_skip("nomic-ai/nomic-embed-text-v1")
    _local_embedder_or_skip("nomic-ai/nomic-embed-text-v1.5")

    DocumentStructure.embedding_model_cls = None
    DocumentStructure.embedding_model_cls_name = None

    legacy = DocumentStructure(
        synthesis_type="test",
        initial_goal="test",
        embedding_model_name="nomic-embed-text-v1",
    )
    v15 = DocumentStructure(
        synthesis_type="test",
        initial_goal="test",
        embedding_model_name="nomic-ai/nomic-embed-text-v1.5",
    )

    assert legacy.embedding_model_name == HUGGINGFACE_EMBEDDING_PATH
    assert v15.embedding_model_name == "nomic-ai/nomic-embed-text-v1.5"
