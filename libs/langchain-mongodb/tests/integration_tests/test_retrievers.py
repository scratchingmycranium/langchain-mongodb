from time import sleep, time
from typing import Generator, List, Any, AsyncGenerator

import pytest
import asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient, AsyncMongoClient
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_mongodb.utils import AsyncCollectionWrapper


from langchain_mongodb import MongoDBAtlasVectorSearch

from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
)

from ..utils import PatchedMongoDBAtlasVectorSearch, IntegrationTestCollection, AsyncCollections

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_retrievers"
COLLECTION_NAME_NESTED = "langchain_test_retrievers_nested"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
PAGE_CONTENT_FIELD = "text"
PAGE_CONTENT_FIELD_NESTED = "title.text"
SEARCH_INDEX_NAME = "text_index"
SEARCH_INDEX_NAME_NESTED = "text_index_nested"

TIMEOUT = 60.0
INTERVAL = 0.5


@pytest.fixture(scope="module")
def example_documents() -> List[Document]:
    return [
        Document(page_content="In 2023, I visited Paris"),
        Document(page_content="In 2022, I visited New York"),
        Document(page_content="In 2021, I visited New Orleans"),
        Document(page_content="Sandwiches are beautiful. Sandwiches are fine."),
    ]


@pytest.fixture(scope="module")
def collection(client: MongoClient, dimensions: int) -> Collection:
    clxn = IntegrationTestCollection(DB_NAME, COLLECTION_NAME).vector_search_index(
        index_name=VECTOR_INDEX_NAME,
        dimensions=dimensions,
        path="embedding",
        similarity="cosine",
    ).fulltext_search_index(
        index_name=SEARCH_INDEX_NAME,
        field=PAGE_CONTENT_FIELD,
    ).sync_collection(client)
        
    return clxn


@pytest.fixture(scope="module")
def collection_nested(client: MongoClient, dimensions: int) -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    clxn = IntegrationTestCollection(DB_NAME, COLLECTION_NAME_NESTED).vector_search_index(
        index_name=VECTOR_INDEX_NAME,
        dimensions=dimensions,
        path="embedding",
        similarity="cosine",
    ).fulltext_search_index(
        index_name=SEARCH_INDEX_NAME_NESTED,
        field=PAGE_CONTENT_FIELD_NESTED,
    ).sync_collection(client)
        
    return clxn


@pytest.fixture(scope="module")
def indexed_vectorstore(
    collection: Collection,
    example_documents: List[Document],
    embedding: Embeddings,
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    vectorstore.add_documents(example_documents)

    yield vectorstore

    vectorstore.collection.delete_many({})


@pytest.fixture(scope="module")
def indexed_nested_vectorstore(
    collection_nested: Collection,
    example_documents: List[Document],
    embedding: Embeddings,
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection_nested,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD_NESTED,
    )

    vectorstore.add_documents(example_documents)

    yield vectorstore

    vectorstore.collection.delete_many({})


def test_vector_retriever(indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test VectorStoreRetriever"""
    retriever = indexed_vectorstore.as_retriever()

    query1 = "When did I visit France?"
    results = retriever.invoke(query1)
    assert len(results) == 4
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_hybrid_retriever(indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        top_k=3,
    )

    query1 = "What did I visit France?"
    results = retriever.invoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_hybrid_retriever_nested(
    indexed_nested_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_nested_vectorstore,
        search_index_name=SEARCH_INDEX_NAME_NESTED,
        top_k=3,
    )

    query1 = "What did I visit France?"
    results = retriever.invoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_fulltext_retriever(
    indexed_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test result of performing fulltext search.

    The Retriever is independent of the VectorStore.
    We use it here only to get the Collection, which we know to be indexed.
    """

    collection: Collection = indexed_vectorstore.collection

    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name=SEARCH_INDEX_NAME,
        search_field=PAGE_CONTENT_FIELD,
    )

    # Wait for the search index to complete.
    search_content = dict(
        index=SEARCH_INDEX_NAME,
        wildcard=dict(query="*", path=PAGE_CONTENT_FIELD, allowAnalyzedField=True),
    )
    n_docs = collection.count_documents({})
    t0 = time()
    while True:
        if (time() - t0) > TIMEOUT:
            raise TimeoutError(
                f"Search index {SEARCH_INDEX_NAME} did not complete in {TIMEOUT}"
            )
        cursor = collection.aggregate([{"$search": search_content}])
        if len(list(cursor)) == n_docs:
            break
        sleep(INTERVAL)

    query = "When was the last time I visited new orleans?"
    results = retriever.invoke(query)
    assert "New Orleans" in results[0].page_content
    assert "score" in results[0].metadata


# Async variants of the tests

@pytest.fixture(
    scope="module",
    params=[
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ]
)
async def async_collection(request, pymongo_async_client: AsyncMongoClient, motor_client: AsyncIOMotorClient, dimensions: int) -> AsyncCollections:
    collection = await (
        IntegrationTestCollection(DB_NAME, COLLECTION_NAME)
        .vector_search_index(
            index_name=VECTOR_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
        )
        .fulltext_search_index(
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
        )
        .async_collection(request, pymongo_async_client, motor_client)
    )
    
    yield collection
    
    await collection.delete_many({})


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ]
)
async def async_collection_nested(request, pymongo_async_client: AsyncMongoClient, motor_client: AsyncIOMotorClient, dimensions: int) -> AsyncCollections:
    collection = await (
        IntegrationTestCollection(DB_NAME, COLLECTION_NAME_NESTED)
        .vector_search_index(
            index_name=VECTOR_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
        )
        .fulltext_search_index(
            index_name=SEARCH_INDEX_NAME_NESTED,
            field=PAGE_CONTENT_FIELD_NESTED,
        )
        .async_collection(request, pymongo_async_client, motor_client)
    )
    
    yield collection
    
    await collection.delete_many({})


@pytest.fixture(scope="module")
async def async_indexed_vectorstore(
    async_collection: AsyncCollections,
    example_documents: List[Document],
    embedding: Embeddings,
) -> AsyncGenerator[MongoDBAtlasVectorSearch, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=async_collection,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    await vectorstore.aadd_documents(example_documents)

    yield vectorstore

    await vectorstore.collection.delete_many({})


@pytest.fixture(scope="module")
async def async_indexed_nested_vectorstore(
    async_collection_nested: AsyncCollections,
    example_documents: List[Document],
    embedding: Embeddings,
) -> AsyncGenerator[MongoDBAtlasVectorSearch, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=async_collection_nested,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD_NESTED,
    )

    await vectorstore.aadd_documents(example_documents)

    yield vectorstore

    await vectorstore.collection.delete_many({})


@pytest.mark.asyncio(loop_scope="session")
async def test_vector_retriever_async(async_indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test VectorStoreRetriever with async client"""
    retriever = async_indexed_vectorstore.as_retriever()

    query1 = "When did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 4
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content


@pytest.mark.asyncio(loop_scope="session")
async def test_hybrid_retriever_async(async_indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever with async client"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=async_indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        top_k=3,
    )

    query1 = "What did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content


@pytest.mark.asyncio(loop_scope="session")
async def test_hybrid_retriever_nested_async(
    async_indexed_nested_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever with async client"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=async_indexed_nested_vectorstore,
        search_index_name=SEARCH_INDEX_NAME_NESTED,
        top_k=3,
    )

    query1 = "What did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content


@pytest.mark.asyncio(loop_scope="session")
async def test_fulltext_retriever_async(
    async_indexed_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test result of performing fulltext search with async client."""

    collection = async_indexed_vectorstore.collection
    collection_wrapper = AsyncCollectionWrapper(collection)

    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name=SEARCH_INDEX_NAME,
        search_field=PAGE_CONTENT_FIELD,
    )

    # Wait for the search index to complete.
    search_content = dict(
        index=SEARCH_INDEX_NAME,
        wildcard=dict(query="*", path=PAGE_CONTENT_FIELD, allowAnalyzedField=True),
    )
    n_docs = await collection.count_documents({})
    t0 = time()
    while True:
        if (time() - t0) > TIMEOUT:
            raise TimeoutError(
                f"Search index {SEARCH_INDEX_NAME} did not complete in {TIMEOUT}"
            )
        count = 0
        async for _ in collection_wrapper.aggregate([{"$search": search_content}]):
            count += 1
        if count == n_docs:
            break
        await asyncio.sleep(INTERVAL)

    query = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query)
    assert "New Orleans" in results[0].page_content
    assert "score" in results[0].metadata
