"""Combine many documents together by recursively reducing them."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, Tuple

from pydantic import Extra

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.docstore.document import Document


class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: List[Document], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""


def _split_list_of_docs(
    docs: List[Document], length_func: Callable, token_max: int, **kwargs: Any
) -> List[List[Document]]:
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError(
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
            if len(_sub_result_docs) == 2:
                raise ValueError(
                    "A single document was so long it could not be combined "
                    "with another document, we cannot handle this."
                )
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list


def _collapse_docs(
    docs: List[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    result = combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


async def _acollapse_docs(
    docs: List[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    result = await combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


class ReduceDocumentsChain(BaseCombineDocumentsChain):
    """Combining documents by recursively reducing them.

    This involves

    - combine_document_chain
    - collapse_document_chain

    `combine_document_chain` is ALWAYS provided. This is final chain that is called.
    We pass all previous results to this chain, and the output of this chain is
    returned as a final result.

    `collapse_document_chain` is used if the documents passed in are too many to all
    be passed to `combine_document_chain` in one go. In this case,
    `collapse_document_chain` is called recursively on as big of groups of documents
    as are allowed.
    """

    combine_document_chain: BaseCombineDocumentsChain
    """Final chain to call to combine documents.
    This is typically a StuffDocumentsChain."""
    collapse_document_chain: Optional[BaseCombineDocumentsChain] = None
    """Chain to use to collapse documents if needed until they can all fit.
    If None, will use the combine_document_chain.
    This is typically a StuffDocumentsChain."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def _collapse_chain(self) -> BaseCombineDocumentsChain:
        if self.collapse_document_chain is not None:
            return self.collapse_document_chain
        else:
            return self.combine_document_chain

    def combine_docs(
        self,
        docs: List[Document],
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine multiple documents recursively.

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        result_docs, extra_return_dict = self._collapse(
            docs, token_max, callbacks=callbacks, **kwargs
        )
        return self.combine_document_chain.combine_docs(
            input_documents=result_docs, callbacks=callbacks, **kwargs
        )

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine multiple documents recursively.

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        result_docs, extra_return_dict = await self._acollapse(
            docs, callbacks=callbacks, **kwargs
        )
        return await self.combine_document_chain.acombine_docs(
            input_documents=result_docs, callbacks=callbacks, **kwargs
        )

    def _collapse(
        self,
        docs: List[Document],
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[List[Document], dict]:
        result_docs = docs
        length_func = self.combine_document_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
            return self._collapse_chain.run(
                input_documents=docs, callbacks=callbacks, **kwargs
            )

        while num_tokens is not None and num_tokens > token_max:
            new_result_doc_list = _split_list_of_docs(
                result_docs, length_func, token_max, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = _collapse_docs(docs, _collapse_docs_func, **kwargs)
                result_docs.append(new_doc)
            num_tokens = length_func(result_docs, **kwargs)
        return result_docs, {}

    async def _acollapse(
        self,
        docs: List[Document],
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Tuple[List[Document], dict]:
        result_docs = docs
        length_func = self.combine_document_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        async def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
            return await self._collapse_chain.arun(
                input_documents=docs, callbacks=callbacks, **kwargs
            )

        while num_tokens is not None and num_tokens > token_max:
            new_result_doc_list = _split_list_of_docs(
                result_docs, length_func, token_max, **kwargs
            )
            result_docs = []
            for docs in new_result_doc_list:
                new_doc = _collapse_docs(docs, _collapse_docs_func, **kwargs)
                result_docs.append(new_doc)
            num_tokens = length_func(result_docs, **kwargs)
        return result_docs, {}

    @property
    def _chain_type(self) -> str:
        return "reduce_documents_chain"
