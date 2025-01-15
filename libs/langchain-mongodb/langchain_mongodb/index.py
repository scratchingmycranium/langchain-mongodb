"""Search Index Commands"""

import logging
import asyncio
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional, Awaitable

from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from langchain_mongodb.utils import AsyncCollectionWrapper, AsyncCollections

logger = logging.getLogger(__file__)


def _vector_search_index_definition(
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    fields = [
        {
            "numDimensions": dimensions,
            "path": path,
            "similarity": similarity,
            "type": "vector",
        },
    ]
    if filters:
        for field in filters:
            fields.append({"type": "filter", "path": field})
    definition = {"fields": fields}
    definition.update(kwargs)
    return definition


def create_vector_search_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[str]] = None,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Experimental Utility function to create a vector search index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index
        filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    result = collection.create_search_index(
        SearchIndexModel(
            definition=_vector_search_index_definition(
                dimensions=dimensions,
                path=path,
                similarity=similarity,
                filters=filters,
                **kwargs,
            ),
            name=index_name,
            type="vectorSearch",
        )
    )

    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: _is_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)

async def acreate_vector_search_index(
    collection: Any,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[str]] = None,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Async version of create_vector_search_index"""
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    result = await collection.create_search_index(
        SearchIndexModel(
            definition=_vector_search_index_definition(
                dimensions=dimensions,
                path=path,
                similarity=similarity,
                filters=filters,
                **kwargs,
            ),
            name=index_name,
            type="vectorSearch",
        )
    )

    if wait_until_complete:
        await _await_for_predicate(
            predicate=lambda: _ais_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)

def drop_vector_search_index(
    collection: Collection,
    index_name: str,
    *,
    wait_until_complete: Optional[float] = None,
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
    collection.drop_search_index(index_name)
    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: len(list(collection.list_search_indexes())) == 0,
            err=f"Index {index_name} did not drop in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Vector Search index %s.%s dropped", collection.name, index_name)

async def adrop_vector_search_index(
    collection: AsyncCollections,
    index_name: str,
    *,
    wait_until_complete: Optional[float] = None,
) -> None:
    """Drop a created vector search index asynchronously

    Args:
        collection (AsyncCollections): MongoDB Collection with index to be dropped
        index_name (str): Name of the MongoDB index
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
    """
    logger.info(
        "Dropping Search Index %s from Collection: %s", index_name, collection.name
    )
    await collection.drop_search_index(index_name)
    if wait_until_complete:
        collection_wrapper = AsyncCollectionWrapper(collection)
        async def check_indexes() -> bool:
            indexes = await collection_wrapper.list_search_indexes()
            return len(indexes) == 0

        await _await_for_predicate(
            predicate=check_indexes,
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
    filters: Optional[List[str]] = None,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Update a search index.

    Replace the existing index definition with the provided definition.

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index.
        filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info(
        "Updating Search Index %s from Collection: %s", index_name, collection.name
    )
    collection.update_search_index(
        name=index_name,
        definition=_vector_search_index_definition(
            dimensions=dimensions,
            path=path,
            similarity=similarity,
            filters=filters,
            **kwargs,
        ),
    )
    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: _is_index_ready(collection, index_name),
            err=f"Index {index_name} update did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Update succeeded")


async def aupdate_vector_search_index(
    collection: Any,
    index_name: str,
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[str]] = None,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Async version of update_vector_search_index.

    Replace the existing index definition with the provided definition.

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding
        path (str): field with vector embedding
        similarity (str): The similarity score used for the index.
        filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info(
        "Updating Search Index %s from Collection: %s", index_name, collection.name
    )
    await collection.update_search_index(
        name=index_name,
        definition=_vector_search_index_definition(
            dimensions=dimensions,
            path=path,
            similarity=similarity,
            filters=filters,
            **kwargs,
        ),
    )
    if wait_until_complete:
        await _await_for_predicate(
            predicate=lambda: _ais_index_ready(collection, index_name),
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
    for index in collection.list_search_indexes(index_name):
        if index["status"] == "READY":
            return True
    return False


async def _ais_index_ready(collection: AsyncCollections, index_name: str) -> bool:
    """Async version of _is_index_ready"""
    collection_wrapper = AsyncCollectionWrapper(collection)
    indexes = await collection_wrapper.list_search_indexes(index_name)
    for index in indexes:
        if index["status"] == "READY":
            return True
    return False

def _wait_for_predicate(
    predicate: Callable, err: str, timeout: float = 120, interval: float = 0.5
) -> None:
    """Generic to block until the predicate returns true

    Args:
        predicate (Callable[, bool]): A function that returns a boolean value
        err (str): Error message to raise if nothing occurs
        timeout (float, optional): Wait time for predicate. Defaults to TIMEOUT.
        interval (float, optional): Interval to check predicate. Defaults to DELAY.

    Raises:
        TimeoutError: _description_
    """
    start = monotonic()
    while not predicate():
        if monotonic() - start > timeout:
            raise TimeoutError(err)
        sleep(interval)

async def _await_for_predicate(
    predicate: Callable[[], Awaitable[bool]], 
    err: str, 
    timeout: float = 120, 
    interval: float = 0.5
) -> None:
    """Async version of _wait_for_predicate

    Args:
        predicate (Callable[[], Awaitable[bool]]): An async function that returns a boolean value
        err (str): Error message to raise if nothing occurs
        timeout (float, optional): Wait time for predicate. Defaults to 120.
        interval (float, optional): Interval to check predicate. Defaults to 0.5.

    Raises:
        TimeoutError: If the predicate doesn't return True within timeout seconds
    """
    start = monotonic()
    while not await predicate():
        if monotonic() - start > timeout:
            raise TimeoutError(err)
        await asyncio.sleep(interval)

def create_fulltext_search_index(
    collection: Collection,
    index_name: str,
    field: str,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Experimental Utility function to create an Atlas Search index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        field (str): Field to index
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    definition = {
        "mappings": {"dynamic": False, "fields": {field: [{"type": "string"}]}}
    }
    result = collection.create_search_index(
        SearchIndexModel(
            definition=definition,
            name=index_name,
            type="search",
            **kwargs,
        )
    )
    if wait_until_complete:
        _wait_for_predicate(
            predicate=lambda: _is_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)

async def acreate_fulltext_search_index(
    collection: AsyncCollections,
    index_name: str,
    field: str,
    *,
    wait_until_complete: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Async version of create_fulltext_search_index

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        field (str): Field to index
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    definition = {
        "mappings": {"dynamic": False, "fields": {field: [{"type": "string"}]}}
    }
    result = await collection.create_search_index(
        SearchIndexModel(
            definition=definition,
            name=index_name,
            type="search",
            **kwargs,
        )
    )
    if wait_until_complete:
        await _await_for_predicate(
            predicate=lambda: _ais_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)
