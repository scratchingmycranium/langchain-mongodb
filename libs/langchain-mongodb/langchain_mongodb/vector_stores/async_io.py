# /langchain_mongodb/vector_stores/async_io.py
from __future__ import annotations
from typing import AsyncIterable, Any, Dict, List, Optional, Iterable, Tuple
from langchain_mongodb.vector_stores.base import MongoDBAtlasVectorSearchBase, DEFAULT_INSERT_BATCH_SIZE
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_mongodb.utils import (
    oid_to_str,
    str_to_oid,
    make_serializable,
)

class AsyncMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearchBase):
    """
    The asynchronous version of MongoDB Atlas Vector Search.
    """

    def __init__(
        self,
        collection: Any,  # e.g. AsyncIOMotorCollection or AsyncIOMongoCollection
        embedding: Embeddings,
        index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        **kwargs: Any,
    ):
        super().__init__(
            embedding=embedding,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )
        self._collection = collection

    async def _io_insert_many_async(self, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Perform an async insert-many and return string IDs.
        """
        result = await self._collection.insert_many(docs)
        return [oid_to_str(_id) for _id in result.inserted_ids]

    async def _io_delete_many_async(
        self, filter_doc: Dict[str, Any], **kwargs: Any
    ) -> bool:
        result = await self._collection.delete_many(filter_doc, **kwargs)
        return result.acknowledged

    async def _io_aggregate_async(
        self, pipeline: List[Dict[str, Any]]
    ) -> AsyncIterable[Dict[str, Any]]:
        cursor = await self._collection.aggregate(pipeline)
        return cursor  # an async cursor

    def _io_insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        raise TypeError("Sync insert called on Async class. Use async method instead.")

    def _io_delete_many(self, filter_doc: Dict[str, Any], **kwargs: Any) -> bool:
        raise TypeError("Sync delete called on Async class. Use async method instead.")

    def _io_aggregate(self, pipeline: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        raise TypeError("Sync aggregate called on Async class. Use async method instead.")

    # ------------------------------------------------------
    # Public asynchronous methods
    # ------------------------------------------------------
    async def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        text_list = list(texts)
        if metadatas is None:
            meta_list = [{} for _ in text_list]
        else:
            meta_list = list(metadatas)

        result_ids: List[str] = []
        start = 0
        n_texts = len(text_list)
        for end in range(batch_size, n_texts + batch_size, batch_size):
            chunk_texts = text_list[start:end]
            chunk_metas = meta_list[start:end]
            if ids:
                chunk_ids = ids[start:end]
                batch_res = await self._bulk_embed_and_insert_texts(chunk_texts, chunk_metas, chunk_ids)
            else:
                batch_res = await self._bulk_embed_and_insert_texts(chunk_texts, chunk_metas)
            result_ids.extend(batch_res)
            start = end

        return result_ids

    async def _bulk_embed_and_insert_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []
        embeddings = self._io_embed_documents(texts)
        if ids:
            to_insert = [
                {
                    "_id": str_to_oid(i),
                    self._text_key: t,
                    self._embedding_key: emb,
                    **m,
                }
                for i, t, m, emb in zip(ids, texts, metadatas, embeddings)
            ]
        else:
            to_insert = [
                {
                    self._text_key: t,
                    self._embedding_key: emb,
                    **m,
                }
                for t, m, emb in zip(texts, metadatas, embeddings)
            ]
        return await self._io_insert_many_async(to_insert)

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict[str, Any]]] = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_vector = self._io_embed_query(query)
        pipeline = self._build_vector_search_pipeline(
            query_vector,
            k,
            pre_filter,
            post_filter_pipeline,
            oversampling_factor,
            include_embeddings,
            **kwargs,
        )
        # gather async aggregate
        async_cursor = await self._io_aggregate_async(pipeline)
        docs_and_scores: List[Tuple[Document, float]] = []
        async for res in async_cursor:
            text = res.pop(self._text_key, "")
            score = res.pop("score", 0.0)
            make_serializable(res)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))
        return docs_and_scores


    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_scores: bool = False,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        """
        The base VectorStore might declare `similarity_search(...)` as abstract.
        Provide a sync stub that raises TypeError, redirecting to the async version.
        """
        raise TypeError(
            "Use `await AsyncMongoDBAtlasVectorSearch.asimilarity_search(...)` instead."
        )
        
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict[str, Any]]] = None,
        oversampling_factor: int = 10,
        include_scores: bool = False,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self.similarity_search_with_score(
            query,
            k,
            pre_filter,
            post_filter_pipeline,
            oversampling_factor,
            include_embeddings,
            **kwargs,
        )
        if include_scores:
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
        return [doc for doc, _ in docs_and_scores]

    async def delete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> bool:
        filter_doc: Dict[str, Any] = {}
        if ids:
            filter_doc = {"_id": {"$in": [str_to_oid(x) for x in ids]}}
        return await self._delete_many_async(filter_doc, **kwargs)

    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict[str, Any]]] = None,
        oversampling_factor: int = 10,
        **kwargs: Any,
    ) -> List[Document]:
        query_vector = self._io_embed_query(query)
        pipeline = self._build_vector_search_pipeline(
            query_vector,
            fetch_k,
            pre_filter,
            post_filter_pipeline,
            oversampling_factor,
            include_embeddings=True,
            **kwargs,
        )
        async_cursor = await self._io_aggregate_async(pipeline)
        docs_and_scores: List[Tuple[Document, float]] = []
        async for res in async_cursor:
            text = res.pop(self._text_key, "")
            score = res.pop("score", 0.0)
            make_serializable(res)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))
        return self._mmr_select(query_vector, docs_and_scores, k, lambda_mult)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        collection: Any,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncMongoDBAtlasVectorSearch:
        """
        This sync signature is required to satisfy the base abstract method.
        But we do NOT actually implement it here. Instead we raise an error.
        """
        raise TypeError(
            "Use `await AsyncMongoDBAtlasVectorSearch.afrom_texts(...)` instead."
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        collection: Any,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncMongoDBAtlasVectorSearch:
        """
        The actual async version that does the same job as from_texts().
        """
        vs = cls(collection, embedding, **kwargs)
        await vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return vs