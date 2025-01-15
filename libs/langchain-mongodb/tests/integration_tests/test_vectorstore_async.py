# """Test async methods of MongoDBAtlasVectorSearch with different async clients."""

# from __future__ import annotations

# import pytest
# from langchain_core.embeddings import Embeddings
# from motor.motor_asyncio import AsyncIOMotorCollection
# from pymongo import AsyncMongoClient

# from langchain_mongodb.index import acreate_vector_search_index
# from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch, AsyncMockCollection

# DB_NAME = "langchain_test_db"
# COLLECTION_NAME = "langchain_test_vectorstores"
# INDEX_NAME = "langchain-test-index"
# DIMENSIONS = 5


# @pytest.fixture(scope="module")
# async def mock_collection() -> AsyncMockCollection:
#     """Get async mock collection."""
#     collection = AsyncMockCollection()
#     await collection.delete_many({})
#     return collection


# class BaseAsyncVectorStoreTest:
#     """Base class for async vectorstore tests."""

#     @pytest.fixture(scope="module")
#     def embeddings(self) -> Embeddings:
#         """Get test embeddings."""
#         return ConsistentFakeEmbeddings(DIMENSIONS)

#     @pytest.fixture(scope="module")
#     def texts(self) -> list[str]:
#         """Get test texts."""
#         return [
#             "Dogs are tough.",
#             "Cats have fluff.",
#             "What is a sandwich?",
#             "That fence is purple.",
#         ]

#     @pytest.fixture(scope="module")
#     def metadatas(self) -> list[dict]:
#         """Get test metadata."""
#         return [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]

#     async def setup_collection(self, collection) -> None:
#         """Setup collection with vector search index."""
#         # Check if index exists by getting list of indexes
#         existing_indexes = []
#         async for index in await collection.list_search_indexes():
#             existing_indexes.append(index["name"])
            
#         if INDEX_NAME not in existing_indexes:
#             await acreate_vector_search_index(
#                 collection=collection,
#                 index_name=INDEX_NAME,
#                 dimensions=DIMENSIONS,
#                 path="embedding",
#                 similarity="cosine",
#                 wait_until_complete=60,
#             )
#         await collection.delete_many({})

#     @pytest.mark.asyncio
#     async def test_add_texts(
#         self, collection, embeddings: Embeddings, texts: list[str], metadatas: list[dict]
#     ) -> None:
#         """Test adding texts asynchronously."""
#         await self.setup_collection(collection)
#         vectorstore = PatchedMongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=INDEX_NAME,
#         )
        
#         # Add texts and verify
#         ids = await vectorstore.aadd_texts(texts=texts, metadatas=metadatas)
#         assert len(ids) == len(texts)

#     @pytest.mark.asyncio
#     async def test_similarity_search(
#         self, collection, embeddings: Embeddings, texts: list[str], metadatas: list[dict]
#     ) -> None:
#         """Test async similarity search."""
#         await self.setup_collection(collection)
#         vectorstore = PatchedMongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=INDEX_NAME,
#         )
        
#         # Add documents first
#         await vectorstore.aadd_texts(texts=texts, metadatas=metadatas)
        
#         # Test basic search
#         results = await vectorstore.asimilarity_search("sandwich", k=1)
#         assert len(results) == 1
#         assert "sandwich" in results[0].page_content.lower()

#         # Test search with score
#         results_with_score = await vectorstore.asimilarity_search_with_score("sandwich", k=1)
#         assert len(results_with_score) == 1
#         doc, score = results_with_score[0]
#         assert "sandwich" in doc.page_content.lower()
#         assert isinstance(score, float)

#     @pytest.mark.asyncio
#     async def test_mmr_search(
#         self, collection, embeddings: Embeddings
#     ) -> None:
#         """Test async maximal marginal relevance search."""
#         await self.setup_collection(collection)
#         vectorstore = PatchedMongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=INDEX_NAME,
#         )
        
#         # Add specific texts for MMR
#         texts = ["foo", "foo", "fou", "foy"]
#         await vectorstore.aadd_texts(texts)
        
#         # Test MMR search
#         results = await vectorstore.amax_marginal_relevance_search(
#             "foo", k=4, lambda_mult=0.1
#         )
#         assert len(results) == len(texts)
#         assert results[0].page_content == "foo"
#         assert results[1].page_content != "foo"

#     @pytest.mark.asyncio
#     async def test_delete(
#         self, collection, embeddings: Embeddings, texts: list[str]
#     ) -> None:
#         """Test async deletion."""
#         await self.setup_collection(collection)
#         vectorstore = PatchedMongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=INDEX_NAME,
#         )
        
#         # Add and then delete texts
#         ids = await vectorstore.aadd_texts(texts=texts)
#         assert len(ids) == len(texts)
        
#         # Delete specific documents
#         success = await vectorstore.adelete(ids=ids[:2])
#         assert success is True
        
#         # Verify deletion
#         results = await vectorstore.asimilarity_search("", k=10)
#         assert len(results) == len(texts) - 2


# class TestMotorVectorStore(BaseAsyncVectorStoreTest):
#     """Test async vectorstore operations using Motor client."""

#     @pytest.fixture(scope="module")
#     async def collection(self, mock_collection) -> AsyncMockCollection:
#         """Get Motor async collection."""
#         return mock_collection


# class TestPyMongoAsyncVectorStore(BaseAsyncVectorStoreTest):
#     """Test async vectorstore operations using PyMongo's beta async client."""

#     @pytest.fixture(scope="module")
#     async def collection(self, mock_collection) -> AsyncMockCollection:
#         """Get PyMongo async collection."""
#         return mock_collection 