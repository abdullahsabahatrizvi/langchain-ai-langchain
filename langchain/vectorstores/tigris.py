import itertools
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VST

if TYPE_CHECKING:
    from tigrisdb import TigrisClient
    from tigrisdb import VectorStore as TigrisVectorStore
    from tigrisdb.types.filters import Filter as TigrisFilter
    from tigrisdb.types.vector import Document as TigrisDocument


class Tigris(VectorStore):
    def __init__(self, client: TigrisClient, embeddings: Embeddings, index_name: str):
        """Initialize Tigris vector store"""
        try:
            import tigrisdb
        except ImportError:
            raise ValueError(
                "Could not import tigrisdb python package. Please install it with `pip install tigrisdb`"
            )

        self._embed_fn = embeddings
        self._vector_store = TigrisVectorStore(client.get_search(), index_name)

    @property
    def search_index(self) -> TigrisVectorStore:
        return self._vector_store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids for documents. Ids will be autogenerated if not provided.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        docs = self._prep_docs(texts, metadatas, ids)
        result = self.search_index.add_documents(docs)
        return [r.id for r in result]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[TigrisFilter] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        docs_with_scores = self.similarity_search_with_score(query, k, filter)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[TigrisFilter] = None,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[TigrisFilter]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        vector = self._embed_fn.embed_query(query)
        result = self.search_index.similarity_search(
            vector=vector, k=k, filter_by=filter
        )
        docs: List[Tuple[Document, float]] = []
        for r in result:
            docs.append(
                (
                    Document(
                        page_content=r.doc["text"], metadata=r.doc.get("metadata")
                    ),
                    r.score,
                )
            )
        return docs

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        client: Optional[TigrisClient] = None,
        index_name: Optional[str] = None,
        **kwargs: Any
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        if not index_name:
            raise ValueError("`index_name` is required")

        if not client:
            client = TigrisClient()
        store = cls(client, embedding, index_name)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    def _prep_docs(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        ids: Optional[List[str]],
    ) -> List[TigrisDocument]:
        embeddings: List[List[float]] = self._embed_fn.embed_documents(list(texts))
        docs: List[TigrisDocument] = []
        for t, m, e, _id in itertools.zip_longest(
            texts, metadatas or [], embeddings or [], ids or []
        ):
            doc: TigrisDocument = {
                "text": t,
                "embeddings": e or [],
                "metadata": m or {},
            }
            if _id:
                doc["id"] = _id
            docs.append(doc)
        return docs
