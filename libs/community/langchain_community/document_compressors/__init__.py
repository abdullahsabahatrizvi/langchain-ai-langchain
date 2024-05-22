import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.document_compressors.flashrank_rerank import (
        FlashrankRerank,
    )
    from langchain_community.document_compressors.jina_rerank import (
        JinaRerank,
    )
    from langchain_community.document_compressors.llmlingua_filter import (
        LLMLinguaCompressor,
    )
    from langchain_community.document_compressors.openvino_rerank import (
        OpenVINOReranker,
    )


_module_lookup = {
    "LLMLinguaCompressor": "langchain_community.document_compressors.llmlingua_filter",
    "OpenVINOReranker": "langchain_community.document_compressors.openvino_rerank",
    "JinaRerank": "langchain_community.document_compressors.jina_rerank",
    "FlashrankRerank": "langchain_community.document_compressors.flashrank_rerank",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["LLMLinguaCompressor", "OpenVINOReranker", "FlashrankRerank", "JinaRerank"]
