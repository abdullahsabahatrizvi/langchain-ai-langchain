"""Retriever that wraps a base retriever and filters the results."""

from typing import Any, List

from pydantic import BaseModel, Extra

from langchain.callbacks.manager import (
    Callbacks,
)
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain.schema import BaseRetriever, Document


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """Retriever that wraps a base retriever and compresses the results."""

    base_compressor: BaseDocumentCompressor
    """Compressor for compressing retrieved documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=callbacks, **kwargs
        )
        if docs:
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=callbacks
            )
            return list(compressed_docs)
        else:
            return []

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.aget_relevant_documents(
            query, callbacks=callbacks, **kwargs
        )
        if docs:
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query, callbacks=callbacks
            )
            return list(compressed_docs)
        else:
            return []
