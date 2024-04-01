"""Filter that uses an LLM to rerank documents listwise and select top-k."""
import logging
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

logger = logging.getLogger(__name__)


class CompressorInput(TypedDict):
    documents: Sequence[Document]
    query: str


_default_system_tmpl = """{context}

Sort the Documents by their relevance to the Query."""
_DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _default_system_tmpl), ("human", "{query}")],
)


def _get_prompt_input(input_: CompressorInput) -> Dict[str, Any]:
    """Return the compression chain input."""
    documents = input_["documents"]
    context = ""
    for index, doc in enumerate(documents):
        context += f"Document ID: {index} ```{doc.page_content}```\n"
    context += f"Documents = [Document ID: 0, ..., Document ID: {len(documents) - 1}]"
    return {"query": input_["query"], "context": context}


def _parse_ranking(results: dict) -> List[Document]:
    ranking = results["ranking"]
    docs = results["documents"]
    return [docs[i] for i in ranking.ranked_document_ids]


class LLMListwiseRerank(BaseDocumentCompressor):
    """Document compressor that uses `Zero-Shot Listwise Document Reranking`.

    Source: https://arxiv.org/pdf/2305.02156.pdf
    """

    reranker: Runnable[CompressorInput, List[Document]]
    """LLM-based reranker to use for filtering documents."""

    top_n: int = 3
    """Number of documents to return."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        results = self.reranker.invoke(
            {"documents": documents, "query": query}, config={"callbacks": callbacks}
        )
        return results[: self.top_n]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> "LLMListwiseRerank":
        """Create a LLMListwiseRerank document compressor from a language model.

        Args:
            llm: The language model to use for filtering. **Must implement
                BaseLanguageModel.with_structured_output().**
            prompt: The prompt to use for the filter.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A LLMListwiseRerank document compressor that uses the given language model.
        """

        if llm.with_structured_output == BaseLanguageModel.with_structured_output:
            raise ValueError(
                f"llm of type {type(llm)} does not implement `with_structured_output`."
            )
        class RankDocuments(BaseModel):
            """Rank the documents by their relevance to the user question.
            Rank from most to least relevant."""

            ranked_document_ids: List[int] = Field(
                ...,
                description=(
                    "The integer IDs of the documents, sorted from most to least "
                    "relevant to the user question."
                ),
            )

        _prompt = prompt if prompt is not None else _DEFAULT_PROMPT
        reranker = RunnablePassthrough.assign(
            ranking=RunnableLambda(_get_prompt_input)
            | _prompt
            | llm.with_structured_output(RankDocuments)
        ) | RunnableLambda(_parse_ranking)
        return cls(reranker=reranker, **kwargs)
