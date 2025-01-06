# /langchain_mongodb/vector_stores/sync_io.py
from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional, Iterable, Tuple
from pymongo.collection import Collection
from langchain_core.documents import Document
from langchain_mongodb.vector_stores.base import MongoDBAtlasVectorSearchBase
from langchain_core.embeddings import Embeddings
from langchain_mongodb.utils import (
    oid_to_str,
    str_to_oid,
)

logger = logging.getLogger(__name__)

class MongoDBAtlasVectorSearch(MongoDBAtlasVectorSearchBase):
  """
  The synchronous version of MongoDB Atlas Vector Search.
  """

  def __init__(
      self,
      collection: Collection,  # sync PyMongo collection
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

  # ------------------------------------------------------
  # Implement the required _io_* methods for synchronous use
  # ------------------------------------------------------
  def _io_insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
      result = self._collection.insert_many(docs)
      return [oid_to_str(_id) for _id in result.inserted_ids]

  def _io_delete_many(
      self, filter_doc: Dict[str, Any], **kwargs: Any
  ) -> bool:
      result = self._collection.delete_many(filter_doc, **kwargs)
      return result.acknowledged

  def _io_aggregate(
      self, pipeline: List[Dict[str, Any]]
  ) -> Iterable[Dict[str, Any]]:
      return self._collection.aggregate(pipeline)

  # ------------------------------------------------------
  # Public synchronous methods
  # ------------------------------------------------------
  def add_texts(
      self,
      texts: Iterable[str],
      metadatas: Optional[List[Dict[str, Any]]] = None,
      ids: Optional[List[str]] = None,
      batch_size: Optional[int] = None,
      **kwargs: Any,
  ) -> List[str]:
      """
      Synchronously embed the texts and insert them into Mongo.
      """
      if metadatas and (metadatas[0].get("_id") or metadatas[0].get("id")):
          logger.warning(
              "_id or id key found in metadata. "
              "Please pop them out or provide as separate list."
          )

      texts_list = list(texts)
      if metadatas is None:
          metadatas_list = [{} for _ in texts_list]
      else:
          metadatas_list = list(metadatas)

      result_ids: List[str] = []
      start = 0
      n_texts = len(texts_list)
      
      # If batch_size is not provided, use the default batch size
      batch_size = batch_size or self._insert_batch_size
      
      # Iterate over the texts in batches
      for end in range(batch_size, n_texts + batch_size, batch_size):
          chunk_texts = texts_list[start:end]
          chunk_metadatas = metadatas_list[start:end]
          if ids:
              chunk_ids = ids[start:end]
              result_ids.extend(self._bulk_embed_and_insert_texts(chunk_texts, chunk_metadatas, chunk_ids))
          else:
              result_ids.extend(self._bulk_embed_and_insert_texts(chunk_texts, chunk_metadatas))
          start = end

      return result_ids

  def similarity_search_with_score(
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
      results = self._io_aggregate(pipeline)
      return self._process_aggregate_result(results)

  def similarity_search(
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
      docs_and_scores = self.similarity_search_with_score(
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

  def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
      filter_doc = {}
      if ids:
          oids = [str_to_oid(x) for x in ids]
          filter_doc = {"_id": {"$in": oids}}
      return self._io_delete_many(filter_doc, **kwargs)

  def max_marginal_relevance_search(
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
      results = self._io_aggregate(pipeline)
      docs_and_scores = self._process_aggregate_result(results)
      return self._mmr_select(query_vector, docs_and_scores, k, lambda_mult)

  @classmethod
  def from_texts(
      cls,
      texts: List[str],
      embedding: Embeddings,
      collection: Collection,
      metadatas: Optional[List[Dict]] = None,
      ids: Optional[List[str]] = None,
      **kwargs: Any,
  ) -> MongoDBAtlasVectorSearch:
      vs = cls(collection, embedding, **kwargs)
      vs.add_texts(texts, metadatas=metadatas, ids=ids)
      return vs

  # ------------------------------------------------------
  # Private methods
  # ------------------------------------------------------
  def _bulk_embed_and_insert_texts(
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

      return self._io_insert_many(to_insert)