"""Wrapper around Activeloop Deep Lake."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class DeepLake(VectorStore):
    """Wrapper around Deep Lake, a data lake for deep learning applications.

    We integrated deeplake's similarity search and filtering for fast prototyping,
    Now, it supports Tensor Query Language (TQL) for production use cases
    over billion rows.

    Why Deep Lake?

    - Not only stores embeddings, but also the original data with version control.
    - Serverless, doesn't require another service and can be used with major
        cloud providers (S3, GCS, etc.)
    - More than just a multi-modal vector store. You can use the dataset
        to fine-tune your own LLM models.

    To use, you should have the ``deeplake`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import DeepLake
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = DeepLake("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_DEEPLAKE_PATH = "./deeplake/"

    def __init__(
        self,
        dataset_path: str = _LANGCHAIN_DEFAULT_DEEPLAKE_PATH,
        token: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        read_only: bool = False,
        ingestion_batch_size: int = 1000,
        num_workers: int = 0,
        verbose: bool = True,
        exec_option: str = "python",
        **kwargs: Any,
    ) -> None:
        """Creates an empty DeepLakeVectorStore or loads an existing one if it exists at
        the specified ``path``.

        Examples:
            >>> # Create a vector store with default tensors
            >>> deeplake_vectorstore = DeepLake(
            ...        path = <path_for_storing_Data>,
            ... )
            >>>
            >>> # Create a vector store in the Deep Lake Managed Tensor Database
            >>> data = DeepLake(
            ...        path = "hub://org_id/dataset_name",
            ...        exec_option = "tensor_db",
            ... )

        Args:
            dataset_path (str): Path to existing dataset or where you want to create
                a new dataset. Defaults to _LANGCHAIN_DEFAULT_DEEPLAKE_PATH.
            token (str, optional):  Activeloop token, used for fetching credentials
                to the dataset at path if it is a Deep Lake dataset. This is optional,
                tokens are normally autogenerated.
            embedding_function (str, optional): Function that will be used to convert
                either documents or query.
            read_only (bool): Whether to open dataset in read-only mode. Defaults to
                False.
            ingestion_batch_size (int): During data ingestion, data is divided into
                batches. The batch size represents the size of each batch.
                Defaults to 1000.
            num_workers (int): Number of workers to use during data ingestion.
                Defaults to 0.
            verbose (bool): Whether to print dataset summary after each operation.
                Defaults to True.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                It could be either "python", "compute_engine" or "tensor_db".
                Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and
                    can be used for data stored anywhere. WARNING: using this option
                    with big datasets is discouraged because it can lead to memory
                    issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    runtime = {"db_engine": True} during dataset creation.
            **kwargs: Other optional keyword arguments.

        Raises:
            ValueError: If some condition is not met. <Please replace with your specific condition>
        """
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        self.verbose = verbose

        try:
            from deeplake.core.vectorstore import DeepLakeVectorStore
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )

        self.dataset_path = dataset_path

        self.vectorstore = DeepLakeVectorStore(
            path=self.dataset_path,
            embedding_function=embedding_function,
            read_only=read_only,
            token=token,
            exec_option=exec_option,
            verbose=verbose,
            **kwargs,
        )

        self._embedding_function = embedding_function
        self._id_tensor_name = "ids" if "ids" in self.vectorstore.tensors() else "id"

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Examples:
            >>> ids = deeplake_vectorstore.add_texts(
            ...     texts = <list_of_texts>,
            ...     metadatas = <list_of_metadata_jsons>,
            ...     ids = <list_of_ids>,
            ... )

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
            **kwargs: other optional keyword arguments.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        kwargs = {}
        if ids:
            if self._id_tensor_name == "ids":  # for backwards compatibility
                kwargs["ids"] = ids
            else:
                kwargs["id"] = ids
        
        if metadatas is None:
            metadatas = [{}]*len(texts)
        
        return self.vectorstore.add(
            text=texts,
            metadata=metadatas,
            embedding_data=texts,
            embedding_tensor="embedding",
            embedding_function=kwargs.get("embedding_function")
            or self._embedding_function.embed_documents,  # type: ignore
            return_ids=True,
            **kwargs,
        )

    def _search_tql(
        self,
        tql_query: str,
        exec_option: Optional[str] = None,
        return_score: bool = False,
    ) -> Any[List[Document], List[Tuple[Document, float]]]:
        """Function for performing tql_search.

        Args:
            tql_query (str, optional): TQL Query string for direct evaluation, without
                application of additional filters or vector search. Available only for
                `compute_engine` and `tensor_db`.
            exec_option (str, optional): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    runtime = {"db_engine": True} during dataset creation.
            return_score (bool): whether to return score with document. Defaults to False.

        Returns:
            List[Document] - A list of documents

        Raises:
            ValueError - if return_score is true.
        """
        result = self.vectorstore.search(
            query=tql_query,
            exec_option=exec_option,
        )
        metadatas = result["metadata"]
        texts = result["text"]

        docs = [
            Document(
                page_content=text,
                metadata=metadata,
            )
            for text, metadata in zip(texts, metadatas)
        ]

        if return_score:
            raise ValueError("scores can't be returned with tql search")

        return docs

    def _search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        embedding_function: Optional[Callable] = None,
        k: int = 4,
        distance_metric: str = "L2",
        use_maximal_marginal_relevance: bool = False,
        fetch_k: Optional[int] = 20,
        filter: Optional[Union[Dict, Callable]] = None,
        return_score: bool = False,
        exec_option: Optional[str] = None,
        **kwargs: Any,
    ) -> Any[List[Document], List[Tuple[Document, float]]]:
        """Return docs most similar to query.

        Args:
            query (str, optional): Text to look up documents similar to.
            tql_query (str, optional): TQL Query string for direct evaluation, without
                application of additional filters or vector search. Available only for
                `compute_engine` and `tensor_db`.
            embedding (Union[List[float], np.ndarray], optional): embedding representation of the query.
                Defaults to None.
            embedding_function (Callable, optional): embedding function that is used to convert `query`
                into embedding.
            k (int): Number of Documents to return. Defaults to 4.
            distance_metric (str): `L2` for Euclidean, `L1` for Nuclear,
                `max` L-infinity distance, `cos` for cosine similarity,
                'dot' for dot product. Defaults to `L2`.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the
                embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis
                    (a sample must satisfy all key-value filters to be True)
                    Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
            use_maximal_marginal_relevance (bool): Whether to use maximal marginal relevance.
                Defaults to False.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            return_score (bool): Whether to return the score. Defaults to False.
            exec_option (str, optional): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    runtime = {"db_engine": True} during dataset creation.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by the specified distance metric,
            if return_score True, return a tuple of (Document, score)

        Raises:
            ValueError: if both `embedding` and `embedding_function` are not specified.
        """
        if kwargs.get("tql_query"):
            return self._search_tql(
                tql_query=kwargs["tql_query"],
                exec_option=exec_option,
                return_score=return_score,
            )

        embedding_function = embedding_function or self._embedding_function  # type: ignore

        if embedding is None:
            if embedding_function is None:
                raise ValueError(
                    "Either `embedding` or `embedding_function` needs to be specified."
                )

            embedding = None
            if query:
                embedding = embedding_function.embed_query(query)  # type: ignore

        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            if len(embedding.shape) > 1:
                embedding = embedding[0]

        result = self.vectorstore.search(
            embedding=embedding,
            k=fetch_k if use_maximal_marginal_relevance else k,
            distance_metric=distance_metric,
            filter=filter,
            exec_option=exec_option,
            return_tensors=["embedding", "metadata", "text"],
        )

        scores = result["score"]
        embeddings = result["embedding"]
        metadatas = result["metadata"]
        texts = result["text"]

        if use_maximal_marginal_relevance:
            lambda_mult = kwargs.get("lambda_mult", 0.5)
            indices = maximal_marginal_relevance(  # type: ignore
                embedding,  # type: ignore
                embeddings,
                k=min(k, len(texts)),
                lambda_mult=lambda_mult,
            )

            scores = [scores[i] for i in indices]
            texts = [texts[i] for i in indices]
            metadatas = [metadatas[i] for i in indices]

        docs = [
            Document(
                page_content=text,
                metadata=metadata,
            )
            for text, metadata in zip(texts, metadatas)
        ]

        if return_score:
            return [(doc, score) for doc, score in zip(docs, scores)]

        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Examples:
            >>> # Search using an embedding
            >>> data = vector_store.similarity_search(
            ...        query = <your_query>,
            ...        k = <number_of_items_to_return>,
            ...        exec_option = <preferred_exec_option>,
            ... )
            >>> # in case if you want to run tql search:
            >>> data = vector_store.tql_search(
            ...        tql_query="SELECT * WHERE id == <some_docs_id>,
            ...        exec_option="compute_engine",
            ... )

        Args:
            k (int): Number of Documents to return.
                Defaults to 4.
            query (str): Text to look up documents similar to.
            embedding (Callable): Embedding function to use.Defaults to None.
            k (int): Number of Documents to return. Defaults to 4.
            distance_metric (str): `L2` for Euclidean, `L1` for Nuclear, `max`
                L-infinity distance, `cos` for cosine similarity, 'dot' for dot product
                Defaults to `L2`.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the
                embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis
                    (a sample must satisfy all key-value filters to be True)
                    `Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}`
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
                Defaults to None.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    `runtime = {"db_engine": True}` during dataset creation.

        Returns:
            List[Document]: List of Documents most similar to the query vector.
        """
        return self._search(
            query=query,
            k=k,
            use_maximal_marginal_relevance=False,
            return_score=False,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: Union[List[float], np.ndarray],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Examples:
            >>> # Search using an embedding
            >>> data = vector_store.similarity_search_by_vector(
            ...        embedding = <your_embedding>,
            ...        k = <number_of_items_to_return>,
            ...        exec_option = <preferred_exec_option>,
            ... )

        Args:
            embedding (Union[List[float], np.ndarray]): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            distance_metric (str): `L2` for Euclidean, `L1` for Nuclear, `max`
                L-infinity distance, `cos` for cosine similarity, 'dot' for dot product
                Defaults to `L2`.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the
                embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis
                    (a sample must satisfy all key-value filters to be True)
                    Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
                Defaults to None.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    `runtime = {"db_engine": True}` during dataset creation.

        Returns:
            List of Documents most similar to the query vector.
        """
        return self._search(
            embedding=embedding,
            k=k,
            use_maximal_marginal_relevance=False,
            return_score=False,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Deep Lake with distance returned.

        Examples:
        >>> # Search using an embedding
        >>> data = vector_store.similarity_search_with_score(
        ...        query = <your_query>,
        ...        embedding=<your_embedding_function>
        ...        k = <number_of_items_to_return>,
        ...        exec_option = <preferred_exec_option>,
        ... )

        Args:
            query (str): Query text to search for.
            distance_metric: `L2` for Euclidean, `L1` for Nuclear, `max` L-infinity
                distance, `cos` for cosine similarity, 'dot' for dot product.
                Defaults to `L2`.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            embedding_function (Callable): Embedding function to use.Defaults to None.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    `runtime = {"db_engine": True}` during dataset creation.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        return self._search(
            query=query,
            k=k,
            return_score=True,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        exec_option: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Examples:
        >>> # Search using an embedding
        >>> data = vector_store.max_marginal_relevance_search_by_vector(
        ...        embedding = <your_embedding>,
        ...        fetch_k = <number_of_elements_to_fetch_before_max_marginal_relevance_search>
        ...        k = <number_of_items_to_return>,
        ...        exec_option = <preferred_exec_option>,
        ... )

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    `runtime = {"db_engine": True}` during dataset creation.
            **kwargs - Additional keyword arguments.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return self._search(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            use_maximal_marginal_relevance=True,
            lambda_mult=lambda_mult,
            exec_option=exec_option,
            **kwargs,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        exec_option: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Examples:
        >>> # Search using an embedding
        >>> data = vector_store.max_marginal_relevance_search(
        ...        query = <query_to_search>,
        ...        embedding_function = <embedding_function_for_query>,
        ...        k = <number_of_items_to_return>,
        ...        exec_option = <preferred_exec_option>,
        ... )

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            embedding (Optional[Callable]): Embedding function to use. Defaults to None.
            exec_option (str): DeepLakeVectorStore supports 3 ways to perform searching.
                    It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be
                    used for data stored anywhere. WARNING: using this option with big datasets
                    is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute
                    Engine that runs on the client and can be used for any data stored in or
                    connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is
                    responsible for storage and query execution. Only available for data stored in
                    the Deep Lake Managed Database. Store datasets in this database by specifying
                    `runtime = {"db_engine": True}` during dataset creation.
            **kwargs: Additional keyword arguments

        Returns:
            List of Documents selected by maximal marginal relevance.

        Raises:
            ValueError: when MRR search is turned on while embedding function is
            not specified.
        """
        embedding_function = kwargs.get("embedding") or self._embedding_function
        if embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on `creation` "
                "or during add call."
            )
        return self._search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            use_maximal_marginal_relevance=True,
            lambda_mult=lambda_mult,
            exec_option=exec_option,
            embedding_function=embedding_function,  # type: ignore
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        dataset_path: str = _LANGCHAIN_DEFAULT_DEEPLAKE_PATH,
        **kwargs: Any,
    ) -> DeepLake:
        """Create a Deep Lake dataset from a raw documents.

        If a dataset_path is specified, the dataset will be persisted in that location,
        otherwise by default at `./deeplake`

        Examples:
        >>> # Search using an embedding
        >>> vector_store = DeepLake.from_texts(
        ...        texts = <the_texts_that_you_want_to_embed>,
        ...        embedding_function = <embedding_function_for_query>,
        ...        k = <number_of_items_to_return>,
        ...        exec_option = <preferred_exec_option>,
        ... )

        Args:
            dataset_path (str): - The full path to the dataset. Can be:
                - Deep Lake cloud path of the form ``hub://username/dataset_name``.
                    To write to Deep Lake cloud datasets,
                    ensure that you are logged in to Deep Lake
                    (use 'activeloop login' from command line)
                - AWS S3 path of the form ``s3://bucketname/path/to/dataset``.
                    Credentials are required in either the environment
                - Google Cloud Storage path of the form
                    ``gcs://bucketname/path/to/dataset`` Credentials are required
                    in either the environment
                - Local file system path of the form ``./path/to/dataset`` or
                    ``~/path/to/dataset`` or ``path/to/dataset``.
                - In-memory path of the form ``mem://path/to/dataset`` which doesn't
                    save the dataset, but keeps it in memory instead.
                    Should be used only for testing as it does not persist.
            texts (List[Document]): List of documents to add.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
                Note, in other places, it is called embedding_function.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            DeepLake: Deep Lake dataset.
        """
        if kwargs.get("embedding"):
            raise ValueError(
                "using embedding as embedidng_functions is deprecated. "
                "Please use `embedding_function` instead."
            )

        deeplake_dataset = cls(
            dataset_path=dataset_path, embedding_function=embedding, **kwargs
        )
        deeplake_dataset.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding_function=embedding.embed_documents,  # type: ignore
        )
        return deeplake_dataset

    def delete(
        self,
        ids: Any[List[str], None] = None,
        filter: Any[Dict[str, str], None] = None,
        delete_all: Any[bool, None] = None,
    ) -> bool:
        """Delete the entities in the dataset

        Args:
            ids (Optional[List[str]], optional): The document_ids to delete.
                Defaults to None.
            filter (Optional[Dict[str, str]], optional): The filter to delete by.
                Defaults to None.
            delete_all (Optional[bool], optional): Whether to drop the dataset.
                Defaults to None.
        """
        self.vectorstore.delete(
            ids=ids,
            filter=filter,
            delete_all=delete_all,
        )

        return True

    @classmethod
    def force_delete_by_path(cls, path: str) -> None:
        """Force delete dataset by path"""
        try:
            import deeplake
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        deeplake.delete(path, large_ok=True, force=True)

    def delete_dataset(self) -> None:
        """Delete the collection."""
        self.delete(delete_all=True)
