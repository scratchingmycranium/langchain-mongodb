from json import dumps, loads
from typing import Any, Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch

from ..utils import ConsistentFakeEmbeddings, AsyncMockCollection

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def get_collection() -> AsyncMockCollection:
    return AsyncMockCollection()

@pytest.fixture()
def collection() -> AsyncMockCollection:
    return get_collection()


@pytest.fixture(scope="module")
def embedding_openai() -> Embeddings:
    return ConsistentFakeEmbeddings()


def test_initialization(collection: AsyncMockCollection, embedding_openai: Embeddings) -> None:
    """Test initialization of vector store class"""
    assert MongoDBAtlasVectorSearch(collection, embedding_openai)


async def test_init_from_texts(collection: AsyncMockCollection, embedding_openai: Embeddings) -> None:
    """Test from_texts operation on an empty list"""
    assert await MongoDBAtlasVectorSearch.afrom_texts(
        [], embedding_openai, collection=collection
    )


class TestMongoDBAtlasVectorSearch:
    @classmethod
    async def setup_class(cls) -> None:
        # ensure the test collection is empty
        collection = get_collection()
        assert await collection.count_documents({}) == 0  # type: ignore[index]

    @classmethod
    async def teardown_class(cls) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        await collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    async def setup(self) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        await collection.delete_many({})  # type: ignore[index]

    @pytest.mark.asyncio
    async def _validate_search(
        self,
        vectorstore: MongoDBAtlasVectorSearch,
        collection: AsyncMockCollection,
        search_term: str = "sandwich",
        page_content: str = "What is a sandwich?",
        metadata: Optional[Any] = 1,
    ) -> None:
        collection._aggregate_result = list(
            filter(
                lambda x: search_term.lower() in x[vectorstore._text_key].lower(),
                collection._data,
            )
        )
        output = await vectorstore.asimilarity_search(search_term, k=1)
        assert output[0].page_content == page_content
        assert output[0].metadata.get("c") == metadata
        # Validate the ObjectId provided is json serializable
        assert loads(dumps(output[0].page_content)) == output[0].page_content
        assert loads(dumps(output[0].metadata)) == output[0].metadata
        assert isinstance(output[0].metadata["_id"], str)

    @pytest.mark.asyncio
    async def test_from_documents(
        self, embedding_openai: Embeddings, collection: AsyncMockCollection
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = await MongoDBAtlasVectorSearch.afrom_documents(
            documents,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        await self._validate_search(
            vectorstore, collection, metadata=documents[2].metadata["c"]
        )

    @pytest.mark.asyncio
    async def test_from_texts(
        self, embedding_openai: Embeddings, collection: AsyncMockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = await MongoDBAtlasVectorSearch.afrom_texts(
            texts,
            embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        await self._validate_search(vectorstore, collection, metadata=None)

    @pytest.mark.asyncio
    async def test_from_texts_with_metadatas(
        self, embedding_openai: Embeddings, collection: AsyncMockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = await MongoDBAtlasVectorSearch.afrom_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        await self._validate_search(vectorstore, collection, metadata=metadatas[2]["c"])

    @pytest.mark.asyncio
    async def test_from_texts_with_metadatas_and_pre_filter(
        self, embedding_openai: Embeddings, collection: AsyncMockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = await MongoDBAtlasVectorSearch.afrom_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        collection._aggregate_result = list(
            filter(
                lambda x: "sandwich" in x[vectorstore._text_key].lower()
                and x.get("c") < 0,
                collection._data,
            )
        )
        output = await vectorstore.asimilarity_search(
            "Sandwich", k=1, pre_filter={"range": {"lte": 0, "path": "c"}}
        )
        assert output == []

    @pytest.mark.asyncio
    async def test_mmr(
        self, embedding_openai: Embeddings, collection: AsyncMockCollection
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = await MongoDBAtlasVectorSearch.afrom_texts(
            texts,
            embedding=embedding_openai,
            collection=collection,
            vector_index_name=INDEX_NAME,
        )
        query = "foo"
        await self._validate_search(
            vectorstore,
            collection,
            search_term=query[0:2],
            page_content=query,
            metadata=None,
        )
        output = await vectorstore.amax_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"