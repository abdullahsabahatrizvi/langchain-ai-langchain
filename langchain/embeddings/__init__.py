"""Wrappers around embedding modules."""
import logging
from typing import Any

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.embeddings.fake import FakeEmbeddings
from langchain.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.embeddings.jina import JinaEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sagemaker_endpoint import SagemakerEndpointEmbeddings
from langchain.embeddings.self_hosted import SelfHostedEmbeddings
from langchain.embeddings.self_hosted_hugging_face import (
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
from langchain.embeddings.tensorflow_hub import TensorflowHubEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAIEmbeddings",
    "HuggingFaceEmbeddings",
    "CohereEmbeddings",
    "JinaEmbeddings",
    "HuggingFaceHubEmbeddings",
    "TensorflowHubEmbeddings",
    "SagemakerEndpointEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "FakeEmbeddings",
]


# TODO: this is in here to maintain backwards compatibility
class HypotheticalDocumentEmbedder:
    def __init__(self, *args: Any, **kwargs: Any):
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H(*args, **kwargs)  # type: ignore

    @classmethod
    def from_llm(cls, *args: Any, **kwargs: Any) -> Any:
        logger.warning(
            "Using a deprecated class. Please use "
            "`from langchain.chains import HypotheticalDocumentEmbedder` instead"
        )
        from langchain.chains.hyde.base import HypotheticalDocumentEmbedder as H

        return H.from_llm(*args, **kwargs)
