import logging
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

logger = logging.getLogger(__file__)

_DELAY = 0.5  # Interval between checks for index operations


def _search_index_error_message():
    return (
        f"Search index operations are not currently available on shared clusters, "
        f"such as MO. They require dedicated clusters >= M10. "
        f"You may still perform vector search. "
        f"You simply must set up indexes manually. Follow the instructions here: "
        f"https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/"
    )


def _vector_search_index_definition(
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    return {
        "fields": [
            {
                "numDimensions": dimensions,
                "path": path,
                "similarity": similarity,
                "type": "vector",
            },
            *(filters or []),
        ]
    }


def create_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[Dict[str, str]]] = None,
    wait_until_complete: Optional[float] = None,
) -> None:
    """Experimental Utility function to create a vector search index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index
        filters (List[Dict[str, str]]): additional filters for index definition.
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    try:
        result = collection.create_search_index(
            SearchIndexModel(
                definition=_vector_search_index_definition(
                    dimensions=dimensions,
                    path=path,
                    similarity=similarity,
                    filters=filters,
                ),
                name=index_name,
                type="vectorSearch",
            )
        )
    except OperationFailure as e:
        raise OperationFailure(_search_index_error_message()) from e

    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: _is_index_ready(collection, index_name),
            err=f"Index {index_name} creation did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)


def drop_vector_search_index(
    collection: Collection, index_name: str, wait_until_complete: Optional[float] = None
) -> None:
    """Drop a created vector search index

    Args:
        collection (Collection): MongoDB Collection with index to be dropped
        index_name (str): Name of the MongoDB index
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
    """
    logger.info(
        "Dropping Search Index %s from Collection: %s", index_name, collection.name
    )
    try:
        collection.drop_search_index(index_name)
    except OperationFailure as e:
        if "CommandNotSupported" in str(e):
            raise OperationFailure(_search_index_error_message()) from e
        # else this most likely means an ongoing drop request was made so skip
    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: len(list(collection.list_search_indexes())) == 0,
            err=f"Index {index_name} did not drop in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Vector Search index %s.%s dropped", collection.name, index_name)


def update_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: List[Dict[str, str]],
    wait_until_complete: Optional[float] = None,
) -> None:
    """Update a search index.

    Replace the existing index definition with the provided definition.

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding.
        path (str): field with vector embedding.
        similarity (str): The similarity score used for the index.
        filters (List[Dict[str, str]]): additional filters for index definition.
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
    """

    logger.info(
        "Updating Search Index %s from Collection: %s", index_name, collection.name
    )
    try:
        collection.update_search_index(
            name=index_name,
            definition=_vector_search_index_definition(
                dimensions=dimensions,
                path=path,
                similarity=similarity,
                filters=filters,
            ),
        )
    except OperationFailure as e:
        raise OperationFailure(_search_index_error_message()) from e

    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: _is_index_ready(collection, index_name),
            err=f"Index {index_name} update did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Update succeeded")


def _is_index_ready(collection: Collection, index_name: str) -> bool:
    """Check for the index name in the list of available search indexes to see if the
    specified index is of status READY

    Args:
        collection (Collection): MongoDB Collection to for the search indexes
        index_name (str): Vector Search Index name

    Returns:
        bool : True if the index is present and READY false otherwise
    """
    try:
        search_indexes = collection.list_search_indexes(index_name)
    except OperationFailure as e:
        raise OperationFailure(_search_index_error_message()) from e

    for index in search_indexes:
        if index["type"] == "vectorSearch" and index["status"] == "READY":
            return True
    return False


def _wait_for_predicate(
    predicate: Callable, err: str, timeout: float = 120, interval: float = 0.5
) -> None:
    """Generic to block until the predicate returns true

    Args:
        predicate (Callable[, bool]): A function that returns a boolean value
        err (str): Error message to raise if nothing occurs
        timeout (float, optional): Length of time to wait for predicate. Defaults to TIMEOUT.
        interval (float, optional): Interval to check predicate. Defaults to DELAY.

    Raises:
        TimeoutError: _description_
    """
    start = monotonic()
    while not predicate():
        if monotonic() - start > timeout:
            raise TimeoutError(err)
        sleep(interval)
