# /langchain_mongodb/vector_stores/base.py
from __future__ import annotations

import numpy as np
from typing import (
  Any,
  Dict,
  Iterable,
  List,
  Optional,
  Tuple,
  TypeVar,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# If you have any indexing or pipeline code:
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import (
    make_serializable,
    maximal_marginal_relevance,
)

VST = TypeVar("VST", bound=VectorStore)

DEFAULT_INSERT_BATCH_SIZE = 100_000

class MongoDBAtlasVectorSearchBase(VectorStore):
    """
    Base class that holds all the non-I/O (business) logic:
    - how to build pipelines
    - how to do MMR re-ranking
    - how to handle local data transformations

    I/O methods (insert, delete, aggregate) are left abstract so that
    the subclasses (sync or async) can implement them properly.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        insert_batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ):
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn
        self._insert_batch_size = insert_batch_size
    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _build_vector_search_pipeline(
        self,
        query_vector: List[float],
        k: int,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Builds the aggregation pipeline for vector search (common logic).
        Subclasses still have to `aggregate()` it in either sync or async fashion.
        """
        stage = vector_search_stage(
            query_vector,
            self._embedding_key,
            self._index_name,
            k,
            pre_filter,
            oversampling_factor,
            **kwargs,
        )
        pipeline: List[Dict[str, Any]] = [
            stage,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})
        if post_filter_pipeline:
            pipeline.extend(post_filter_pipeline)
        return pipeline

    def _process_aggregate_result(
        self, results: Iterable[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """
        Convert raw Mongo results into List[(Document, score)].
        """
        docs_and_scores: List[Tuple[Document, float]] = []
        for res in results:
            text = res.pop(self._text_key, "")
            score = res.pop("score", 0.0)
            make_serializable(res)
            docs_and_scores.append((Document(page_content=text, metadata=res), score))
        return docs_and_scores

    def _mmr_select(
        self,
        base_vector: List[float],
        docs_and_scores: List[Tuple[Document, float]],
        k: int,
        lambda_mult: float,
    ) -> List[Document]:
        """
        Runs maximal marginal relevance re-ranking.
        """
        vectors = [doc.metadata[self._embedding_key] for doc, _ in docs_and_scores]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(base_vector),
            np.array(vectors),
            k=k,
            lambda_mult=lambda_mult,
        )
        return [docs_and_scores[i][0] for i in mmr_doc_indexes]

    # -------------------------------------------------------------------
    # Abstract or "unimplemented" I/O methods that children must override
    # -------------------------------------------------------------------

    def _io_insert_many(
        self, docs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert multiple documents. Returns list of inserted IDs as strings.
        Implemented by child classes for sync or async.
        """
        raise NotImplementedError()

    def _io_delete_many(
        self, filter_doc: Dict[str, Any], **kwargs: Any
    ) -> bool:
        """
        Delete documents matching filter_doc. Return True if acknowledged.
        Implemented by child classes for sync or async.
        """
        raise NotImplementedError()

    def _io_aggregate(
        self, pipeline: List[Dict[str, Any]]
    ) -> Iterable[Dict[str, Any]]:
        """
        Aggregate with a pipeline. Return an iterator/iterable over results.
        Implemented by child classes for sync or async.
        """
        raise NotImplementedError()

    def _io_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Possibly override in child classes if your embedding is truly async.
        Otherwise, default to self._embedding.embed_documents() (sync).
        """
        return self._embedding.embed_documents(texts)

    def _io_embed_query(self, query: str) -> List[float]:
        """
        Possibly override in child classes if your embedding is truly async.
        Otherwise, default to self._embedding.embed_query() (sync).
        """
        return self._embedding.embed_query(query)
