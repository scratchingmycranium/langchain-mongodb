"""Test max_marginal_relevance_search."""

from __future__ import annotations

import pytest  # type: ignore[import-not-found]
from typing import Union
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient, AsyncMongoClient
from motor.motor_asyncio import (
    AsyncIOMotorClient,
)
from pymongo.collection import Collection
from ..utils import (
    IntegrationTestCollection,
    AsyncCollections
)

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_vectorstores"
INDEX_NAME = "langchain-test-index-vectorstores"
DIMENSIONS = 5

@pytest.fixture()
def collection(client: MongoClient) -> Collection:
    return IntegrationTestCollection(DB_NAME, COLLECTION_NAME) \
        .vector_search_index(
            index_name=INDEX_NAME,
            filters=["c"],
            path="embedding",
            similarity="cosine",
            dimensions=DIMENSIONS
        ) \
        .sync_collection(client)

@pytest.fixture(
    scope="function",
    params=[
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ]
)
async def async_collection(request, pymongo_async_client: AsyncMongoClient, motor_client: AsyncIOMotorClient) -> AsyncCollections:
    collection = await (
        IntegrationTestCollection(DB_NAME, COLLECTION_NAME)
        .vector_search_index(
            index_name=INDEX_NAME,
            filters=["c"],
            path="embedding",
            similarity="cosine",
            dimensions=DIMENSIONS
        )
        .async_collection(request, pymongo_async_client, motor_client)
    )
    
    yield collection
    
    await collection.delete_many({})

@pytest.fixture
def embeddings() -> Embeddings:
    return ConsistentFakeEmbeddings(DIMENSIONS)


def test_mmr(embeddings: Embeddings, collection: Collection) -> None:
    texts = ["foo", "foo", "fou", "foy"]
    collection.delete_many({})
    vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME,
    )
    query = "foo"
    output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
    assert len(output) == len(texts)
    assert output[0].page_content == "foo"
    assert output[1].page_content != "foo"

@pytest.mark.asyncio
async def test_mmr_async(embeddings: Embeddings, async_collection: AsyncCollections):
    texts = ["foo", "foo", "fou", "foy"]
    print('debug 0')
    await async_collection.delete_many({})
    print('debug 1')
    vectorstore = await PatchedMongoDBAtlasVectorSearch.afrom_texts(
        texts,
        embedding=embeddings,
        collection=async_collection,
        index_name=INDEX_NAME,
    )
    query = "foo"
    print('debug 2')
    output = await vectorstore.amax_marginal_relevance_search(query, k=10, lambda_mult=0.1)
    print('debug 3')
    assert len(output) == len(texts)
    assert output[0].page_content == "foo"
    assert output[1].page_content != "foo"
