from __future__ import annotations

from copy import deepcopy
from time import monotonic, sleep
import math
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Union, cast, AsyncIterator, AsyncIterable
from bson import ObjectId
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import model_validator
from pymongo.collection import Collection
from pymongo.results import DeleteResult, InsertManyResult
from pymongo import AsyncMongoClient, MongoClient
from pymongo.asynchronous.collection import AsyncCollection
from motor.motor_asyncio import AsyncIOMotorCollection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_mongodb.index import (
    acreate_vector_search_index,
    acreate_fulltext_search_index,
    create_vector_search_index,
    create_fulltext_search_index
)

import asyncio

# Constants for timeouts and intervals
TIMEOUT = 120
INTERVAL = 0.5

# Type alias for async collections from either motor or pymongo
AsyncCollections = Union[AsyncIOMotorClient, Collection]

class PatchedMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = super().bulk_embed_and_insert_texts(texts, metadatas, ids)
        n_docs = self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if (
                len(self.similarity_search("sandwich", k=n_docs, oversampling_factor=1))
                == n_docs
            ):
                return ids_inserted
            else:
                sleep(INTERVAL)
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")

    async def abulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Async patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = await super().abulk_embed_and_insert_texts(texts, metadatas, ids)
        n_docs = await self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            results = await self.asimilarity_search("sandwich", k=n_docs, oversampling_factor=1)
            if len(results) == n_docs:
                return ids_inserted
            else:
                await asyncio.sleep(INTERVAL)
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")


class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [1.0] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = "fake response"
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"key": "fake"}


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def check_queries_required(cls, values: dict) -> dict:
        if values.get("sequential_response") and not values.get("queries"):
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return values

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


class MockCollection(Collection):
    """Mocked Mongo Collection"""

    _aggregate_result: List[Any]
    _insert_result: Optional[InsertManyResult]
    _data: List[Any]
    _simulate_cache_aggregation_query: bool

    def __init__(self) -> None:
        self._data = []
        self._aggregate_result = []
        self._insert_result = None
        self._simulate_cache_aggregation_query = False

    def delete_many(self, *args, **kwargs) -> DeleteResult:  # type: ignore
        old_len = len(self._data)
        self._data = []
        return DeleteResult({"n": old_len}, acknowledged=True)

    def insert_many(self, to_insert: List[Any], *args, **kwargs) -> InsertManyResult:  # type: ignore
        mongodb_inserts = [
            {"_id": ObjectId(), "score": 1, **insert} for insert in to_insert
        ]
        self._data.extend(mongodb_inserts)
        return self._insert_result or InsertManyResult(
            [k["_id"] for k in mongodb_inserts], acknowledged=True
        )

    def insert_one(self, to_insert: Any, *args, **kwargs) -> Any:  # type: ignore
        return self.insert_many([to_insert])

    def find_one(self, find_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # type: ignore
        find = self.find(find_query) or [None]  # type: ignore
        return find[0]

    def find(self, find_query: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:  # type: ignore
        def _is_match(item: Dict[str, Any]) -> bool:
            for key, match_val in find_query.items():
                if item.get(key) != match_val:
                    return False
            return True

        return [document for document in self._data if _is_match(document)]

    def update_one(  # type: ignore
        self,
        find_query: Dict[str, Any],
        options: Dict[str, Any],
        *args: Any,
        upsert=True,
        **kwargs: Any,
    ) -> None:  # type: ignore
        result = self.find_one(find_query)
        set_options = options.get("$set", {})

        if result:
            result.update(set_options)
        elif upsert:
            self._data.append({**find_query, **set_options})

    def _execute_cache_aggregation_query(self, *args, **kwargs) -> List[Dict[str, Any]]:  # type: ignore
        """Helper function only to be used for MongoDBAtlasSemanticCache Testing

        Returns:
            List[Dict[str, Any]]: Aggregation query result
        """
        pipeline: List[Dict[str, Any]] = args[0]
        params = pipeline[0]["$vectorSearch"]
        # Assumes MongoDBAtlasSemanticCache.LLM == "llm_string"
        llm_string = params["filter"][MongoDBAtlasSemanticCache.LLM]["$eq"]

        acc = []
        for document in self._data:
            if (
                document.get("embedding") == embedding
                and document.get(MongoDBAtlasSemanticCache.LLM) == llm_string
            ):
                acc.append(document)
        return acc

    def aggregate(self, *args, **kwargs) -> List[Any]:  # type: ignore
        if self._simulate_cache_aggregation_query:
            return deepcopy(self._execute_cache_aggregation_query(*args, **kwargs))
        return deepcopy(self._aggregate_result)

    def count_documents(self, *args, **kwargs) -> int:  # type: ignore
        return len(self._data)

    def __repr__(self) -> str:
        return "MockCollection"

class MockAsyncCursor:
    def __init__(self, data: List[Dict[str, Any]]):
        self._data = data
        self._index = 0  # To track the cursor index

    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Enable async iteration.
        """
        return self

    async def __anext__(self) -> Dict[str, Any]:
        """
        Provide the next document in the dataset asynchronously.
        """
        await asyncio.sleep(0)  # Simulate async behavior
        if self._index < len(self._data):
            item = self._data[self._index]
            self._index += 1
            return item
        raise StopAsyncIteration

    async def to_list(self, length: int = None) -> List[Dict[str, Any]]:
        """
        Return all documents as a list, optionally limiting the length.
        """
        await asyncio.sleep(0)  # Simulate async behavior
        return self._data[:length] if length else self._data


def _doc_matches_filter(doc: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """
    A basic filter matcher. Real MongoDB supports operators like $gt, $regex, etc.
    You can expand this as needed if your test uses more-complex queries.
    """
    for k, v in flt.items():
        # Very naive approach:
        if k not in doc:
            return False
        
        # If the test code uses an operator, e.g. {"text": {"$regex": ...}}, 
        # you'll need to handle it separately here:
        if isinstance(v, dict):
            # Example for a naive $regex check:
            if "$regex" in v:
                import re
                pattern = re.compile(v["$regex"], flags=0)
                if not pattern.search(doc[k]):
                    return False
            else:
                # Other operators could be handled similarly
                return False
        else:
            # Basic equality check
            if doc[k] != v:
                return False
    return True

def cosine_similarity(vec1, vec2):
    # Basic safe-guard for zero-length
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1) or 1e-9)
    norm2 = math.sqrt(sum(b*b for b in vec2) or 1e-9)
    return dot / (norm1 * norm2)

class AsyncMockCollection:
    """Async Mocked Mongo Collection."""

    def __init__(self) -> None:
        self._data: List[Dict[str, Any]] = []
        # If you need an aggregated result separate from _data, you can store it here:
        self._aggregate_result: List[Dict[str, Any]] = []
        self._simulate_cache_aggregation_query = False

    async def delete_many(self, flt: Dict[str, Any] = None, *args, **kwargs) -> DeleteResult:
        if flt is None:
            flt = {}
        old_len = len(self._data)

        # If it's a simple {"_id": {"$in": [...]}} filter, handle that explicitly:
        in_ids = set()
        # Check if we have {"_id": {"$in": [...]}}, else do a more general fallback
        if "_id" in flt and isinstance(flt["_id"], dict) and "$in" in flt["_id"]:
            in_ids = set(flt["_id"]["$in"])

        new_data = []
        removed = 0

        for doc in self._data:
            # If we do have a set of IDs to remove, check membership:
            if in_ids:
                if doc["_id"] in in_ids:
                    removed += 1
                else:
                    new_data.append(doc)
            else:
                # Fallback to generic matching if not an $in query
                if not _doc_matches_filter(doc, flt):
                    new_data.append(doc)
                else:
                    removed += 1

        self._data = new_data
        return DeleteResult({"n": removed}, acknowledged=True)

    async def insert_many(
        self, to_insert: List[Any], *args, **kwargs
    ) -> InsertManyResult:
        mongodb_inserts = [
            {"_id": ObjectId(), "score": 1, **doc} for doc in to_insert
        ]
        self._data.extend(mongodb_inserts)
        return InsertManyResult(
            inserted_ids=[doc["_id"] for doc in mongodb_inserts],
            acknowledged=True,
        )

    async def insert_one(self, doc: Dict[str, Any], *args, **kwargs) -> InsertManyResult:
        return await self.insert_many([doc])

    async def count_documents(self, flt: Dict[str, Any] = None, **kwargs) -> int:
        if flt is None:
            flt = {}
        return sum(_doc_matches_filter(doc, flt) for doc in self._data)

    def find(self, flt: Dict[str, Any] = None) -> MockAsyncCursor:
        """
        Simulate Motor's find method, returning an async cursor.
        """
        if flt is None:
            flt = {}
        matched_docs = [d for d in self._data if _doc_matches_filter(d, flt)]
        return MockAsyncCursor(matched_docs)

    async def find_one(self, flt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for doc in self._data:
            if _doc_matches_filter(doc, flt):
                return doc
        return None

    async def update_one(
        self,
        find_query: Dict[str, Any],
        update_spec: Dict[str, Any],
        *args: Any,
        upsert=True,
        **kwargs: Any,
    ):
        """
        Patch the first matching doc or insert if upsert=True and no match.
        """
        doc = await self.find_one(find_query)
        set_options = update_spec.get("$set", {})

        if doc:
            doc.update(set_options)
        elif upsert:
            new_doc = {**find_query, **set_options}
            await self.insert_one(new_doc)

    async def aggregate(self, pipeline: List[Dict[str, Any]], *args, **kwargs) -> MockAsyncCursor:
        if self._simulate_cache_aggregation_query:
            return MockAsyncCursor(deepcopy(self._execute_cache_aggregation_query(*args, **kwargs)))
        return MockAsyncCursor(deepcopy(self._aggregate_result))

    def __repr__(self) -> str:
        return "AsyncMockCollection"

class IntegrationTestCollection:
    def __init__(
        self,
        db_name: str = "test_db",
        collection_name: str = "test_collection", 
        filters: List[str] = ["c"],
        path: str = "embedding",
        similarity: str = "cosine",
        dimensions: int = 5,
        timeout: int = 60
    ) -> None:
        self.DB_NAME = db_name
        self.COLLECTION_NAME = collection_name
        self.FILTERS = filters
        self.PATH = path
        self.SIMILARITY = similarity
        self.DIMENSIONS = dimensions
        self.TIMEOUT = timeout
        self._vector_index_config = None
        self._fulltext_index_config = None

    def vector_search_index(
        self,
        index_name: str,
        dimensions: Optional[int] = None,
        path: Optional[str] = None,
        filters: Optional[List[str]] = None,
        similarity: Optional[str] = None,
    ) -> "IntegrationTestCollection":
        self._vector_index_config = {
            "index_name": index_name,
            "dimensions": dimensions or self.DIMENSIONS,
            "path": path or self.PATH,
            "filters": filters or self.FILTERS,
            "similarity": similarity or self.SIMILARITY,
        }
        return self

    def fulltext_search_index(
        self,
        index_name: str,
        field: str = "test_field",
    ) -> "IntegrationTestCollection":
        self._fulltext_index_config = {
            "index_name": index_name,
            "field": field,
        }
        return self

    async def async_collection(self, request, pymongo_async_client: AsyncMongoClient, motor_client: AsyncIOMotorClient) -> AsyncCollections:
        client = pymongo_async_client if request.param == "pymongo" else motor_client
        db = client[self.DB_NAME]
        collection_names = await db.list_collection_names()
        if self.COLLECTION_NAME not in collection_names:
            clxn = await db.create_collection(self.COLLECTION_NAME)
        else:
            clxn = db[self.COLLECTION_NAME]

        await clxn.delete_many({})

        if request.param == "motor":
            cursor = clxn.list_search_indexes()
            indexes = await cursor.to_list(length=None)
        else:
            indexes = await (await clxn.list_search_indexes()).to_list(None)
            
        if self._vector_index_config and not any([self._vector_index_config["index_name"] == ix["name"] for ix in indexes]):
            await acreate_vector_search_index(
                collection=clxn,
                index_name=self._vector_index_config["index_name"],
                dimensions=self._vector_index_config["dimensions"],
                path=self._vector_index_config["path"],
                filters=self._vector_index_config["filters"],
                similarity=self._vector_index_config["similarity"],
                wait_until_complete=self.TIMEOUT,
            )
            
        if self._fulltext_index_config and not any([self._fulltext_index_config["index_name"] == ix["name"] for ix in indexes]):
            await acreate_fulltext_search_index(
                collection=clxn,
                index_name=self._fulltext_index_config["index_name"],
                field=self._fulltext_index_config["field"],
                wait_until_complete=self.TIMEOUT,
            )

        return clxn

    def sync_collection(self, client: MongoClient) -> Collection:
        db = client[self.DB_NAME]
        collection_names = db.list_collection_names()
        if self.COLLECTION_NAME not in collection_names:
            clxn = db.create_collection(self.COLLECTION_NAME)
        else:
            clxn = db[self.COLLECTION_NAME]

        clxn.delete_many({})

        indexes = list(clxn.list_search_indexes())
        
        if self._vector_index_config and not any([self._vector_index_config["index_name"] == ix["name"] for ix in indexes]):
            create_vector_search_index(
                collection=clxn,
                index_name=self._vector_index_config["index_name"],
                dimensions=self._vector_index_config["dimensions"],
                path=self._vector_index_config["path"],
                filters=self._vector_index_config["filters"],
                similarity=self._vector_index_config["similarity"],
                wait_until_complete=self.TIMEOUT,
            )
            
        if self._fulltext_index_config and not any([self._fulltext_index_config["index_name"] == ix["name"] for ix in indexes]):
            create_fulltext_search_index(
                collection=clxn,
                index_name=self._fulltext_index_config["index_name"],
                field=self._fulltext_index_config["field"],
                wait_until_complete=self.TIMEOUT,
            )

        return clxn