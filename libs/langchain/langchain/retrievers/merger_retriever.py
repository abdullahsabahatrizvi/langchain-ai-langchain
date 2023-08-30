from typing import Any, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class MergerRetriever(BaseRetriever):
    """Retriever that merges the results of multiple retrievers."""

    retrievers: List[BaseRetriever]
    """A list of retrievers to merge."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = self.merge_documents(
            query, run_manager, retrievers_kwargs=kwargs
        )

        return merged_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant documents.
        """

        # Merge the results of the retrievers.
        merged_documents = await self.amerge_documents(query, run_manager)

        return merged_documents

    def merge_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        retrievers_kwargs: Any | None = None,
    ) -> List[Document]:
        """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.
            retrievers_kwargs: List containing the kwargs to pass to each retriever.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        if retrievers_kwargs:
            if not isinstance(retrievers_kwargs, list):
                retrievers_kwargs_list = [retrievers_kwargs] * len(self.retrievers)
        else:
            retrievers_kwargs_list = [{}] * len(self.retrievers)
        retriever_docs = [
            retriever.get_relevant_documents(
                query,
                callbacks=run_manager.get_child("retriever_{}".format(i + 1)),
                **retrievers_kwargs_list[i],
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(len(docs) for docs in retriever_docs)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

    async def amerge_documents(
        self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronously merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            await retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child("retriever_{}".format(i + 1))
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        max_docs = max(len(docs) for docs in retriever_docs)
        for i in range(max_docs):
            for retriever, doc in zip(self.retrievers, retriever_docs):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents
