from typing import Generator, List, Optional, AsyncGenerator

import pytest
import asyncio
from pymongo import MongoClient, AsyncMongoClient
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorClient

from langchain_mongodb import MongoDBAtlasVectorSearch, index
from langchain_mongodb.utils import AsyncCollectionWrapper
from ..utils import ConsistentFakeEmbeddings, AsyncCollections, IntegrationTestCollection

DB_NAME = "langchain_test_index_db"
COLLECTION_NAME = "test_index"
VECTOR_INDEX_NAME = "vector_index"

TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture
def collection(client: MongoClient) -> Generator:
    """Depending on uri, this could point to any type of cluster."""
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]
    clxn = client[DB_NAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


def test_search_index_drop_add_delete_commands(collection: Collection) -> None:
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: Optional[List[str]] = None
    wait_until_complete = TIMEOUT

    for index_info in collection.list_search_indexes():
        index.drop_vector_search_index(
            collection, index_info["name"], wait_until_complete=wait_until_complete
        )

    assert len(list(collection.list_search_indexes())) == 0

    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    index.drop_vector_search_index(
        collection, index_name, wait_until_complete=wait_until_complete
    )

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 0


@pytest.mark.skip("collection.update_vector_search_index requires [CLOUDP-275518]")
def test_search_index_update_vector_search_index(collection: Collection) -> None:
    index_name = "INDEX_TO_UPDATE"
    similarity_orig = "cosine"
    similarity_new = "euclidean"

    # Create another index
    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_orig,
        wait_until_complete=TIMEOUT,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_orig

    # Update the index and test new similarity
    index.update_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_new,
        wait_until_complete=TIMEOUT,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_new


def test_vectorstore_create_vector_search_index(collection: Collection) -> None:
    """Tests vectorstore wrapper around index command."""

    # Set up using the index module's api
    if len(list(collection.list_search_indexes())) != 0:
        index.drop_vector_search_index(
            collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
        )

    # Test MongoDBAtlasVectorSearch's API
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=ConsistentFakeEmbeddings(),
        index_name=VECTOR_INDEX_NAME,
    )

    vectorstore.create_vector_search_index(
        dimensions=DIMENSIONS, wait_until_complete=TIMEOUT
    )

    assert index._is_index_ready(collection, VECTOR_INDEX_NAME)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == VECTOR_INDEX_NAME

    # Tear down using the index module's api
    index.drop_vector_search_index(
        collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ]
)
async def async_collection(request, pymongo_async_client: AsyncMongoClient, motor_client: AsyncIOMotorClient) -> AsyncCollections:
    collection = await (
        IntegrationTestCollection(DB_NAME, COLLECTION_NAME)
        .vector_search_index(
            index_name=VECTOR_INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
        )
        .async_collection(request, pymongo_async_client, motor_client)
    )
    
    yield collection
    
    await collection.delete_many({})

@pytest.mark.asyncio(loop_scope="session")
async def test_search_index_drop_add_delete_commands_async(async_collection: AsyncCollections) -> None:
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: Optional[List[str]] = None
    wait_until_complete = TIMEOUT

    collection_wrapper = AsyncCollectionWrapper(async_collection)
    indexes = await collection_wrapper.list_search_indexes()
    for index_info in indexes:
        await index.adrop_vector_search_index(
            async_collection, index_info["name"], wait_until_complete=wait_until_complete
        )

    indexes = await collection_wrapper.list_search_indexes()
    assert len(indexes) == 0

    await index.acreate_vector_search_index(
        collection=async_collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    assert await index._ais_index_ready(async_collection, index_name)
    indexes = await collection_wrapper.list_search_indexes()
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    await index.adrop_vector_search_index(
        async_collection, index_name, wait_until_complete=wait_until_complete
    )

    indexes = await collection_wrapper.list_search_indexes()
    assert len(indexes) == 0


@pytest.mark.skip("collection.update_vector_search_index requires [CLOUDP-275518]")
@pytest.mark.asyncio(loop_scope="session")
async def test_search_index_update_vector_search_index_async(async_collection: AsyncCollections) -> None:
    index_name = "INDEX_TO_UPDATE"
    similarity_orig = "cosine"
    similarity_new = "euclidean"

    # Create another index
    await index.acreate_vector_search_index(
        collection=async_collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_orig,
        wait_until_complete=TIMEOUT,
    )

    assert await index._ais_index_ready(async_collection, index_name)
    indexes = []
    async for idx in async_collection.list_search_indexes():
        indexes.append(idx)
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_orig

    # Update the index and test new similarity
    await index.aupdate_vector_search_index(
        collection=async_collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_new,
        wait_until_complete=TIMEOUT,
    )

    assert await index._ais_index_ready(async_collection, index_name)
    indexes = []
    async for idx in async_collection.list_search_indexes():
        indexes.append(idx)
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_new


@pytest.mark.asyncio(loop_scope="session")
async def test_vectorstore_create_vector_search_index_async(async_collection: AsyncCollections) -> None:
    """Tests vectorstore wrapper around index command with async client."""

    collection_wrapper = AsyncCollectionWrapper(async_collection)
    # Set up using the index module's api
    indexes = await collection_wrapper.list_search_indexes()
    if len(indexes) != 0:
        print("Dropping index")
        await index.adrop_vector_search_index(
            async_collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
        )

    # Test MongoDBAtlasVectorSearch's API
    vectorstore = MongoDBAtlasVectorSearch(
        collection=async_collection,
        embedding=ConsistentFakeEmbeddings(),
        index_name=VECTOR_INDEX_NAME,
    )

    await vectorstore.acreate_vector_search_index(
        dimensions=DIMENSIONS, wait_until_complete=TIMEOUT
    )

    assert await index._ais_index_ready(async_collection, VECTOR_INDEX_NAME)
    indexes = await collection_wrapper.list_search_indexes()
    assert len(indexes) == 1
    assert indexes[0]["name"] == VECTOR_INDEX_NAME

    # Tear down using the index module's api
    await index.adrop_vector_search_index(
        async_collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
    )
