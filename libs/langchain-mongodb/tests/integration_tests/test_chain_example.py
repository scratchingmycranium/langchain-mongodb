"Demonstrates MongoDBAtlasVectorSearch.as_retriever() invoked in a chain" ""

from __future__ import annotations

import os

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import AsyncMongoClient

from langchain_mongodb import index

from ..utils import PatchedMongoDBAtlasVectorSearch, AsyncCollections

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_chain_example"
INDEX_NAME = "langchain-test-chain-example-vector-index"
DIMENSIONS = 1536
TIMEOUT = 60.0
INTERVAL = 0.5


@pytest.fixture
def collection(client: MongoClient) -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if all([INDEX_NAME != ix["name"] for ix in clxn.list_search_indexes()]):
        index.create_vector_search_index(
            collection=clxn,
            index_name=INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
            filters=None,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OpenAI for chat responses.",
)
def test_chain(
    collection: Collection,
    embedding: Embeddings,
) -> None:
    """Demonstrate usage of MongoDBAtlasVectorSearch in a realistic chain

    Follows example in the docs: https://python.langchain.com/docs/how_to/hybrid/

    Requires OpenAI_API_KEY for embedding and chat model.
    Requires INDEX_NAME to have been set up on MONGODB_URI
    """

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=INDEX_NAME,
        text_key="page_content",
    )

    texts = [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
        "In 2019, I visited San Francisco",
        "In 2020, I visited Vancouver",
    ]
    vectorstore.add_texts(texts)

    query = "In the United States, what city did I visit last?"
    # One can do vector search on the vector store, using its various search types.
    k = len(texts)

    store_output = list(vectorstore.similarity_search(query=query, k=k))
    assert len(store_output) == k
    assert isinstance(store_output[0], Document)

    # Unfortunately, the VectorStore output cannot be given to a Chat Model
    # If we wish Chat Model to answer based on our own data,
    # we have to give it the right things to work with.
    # The way that Langchain does this is by piping results along in
    # a Chain: https://python.langchain.com/v0.1/docs/modules/chains/

    # Now, we can turn our VectorStore into something Runnable in a Chain
    # by turning it into a Retriever.
    # For the simple VectorSearch Retriever, we can do this like so.

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=k))

    # This does not do much other than expose our search function
    # as an invoke() method with a a certain API, a Runnable.
    retriever_output = retriever.invoke(query)
    assert len(retriever_output) == len(texts)
    assert retriever_output[0].page_content == store_output[0].page_content

    # To get a natural language response to our question,
    # we need ChatOpenAI, a template to better frame the question as a prompt,
    # and a parser to send the output to a string.
    # Together, these become our Chain!
    # Here goes:

    template = """Answer the question based only on the following context.
     Answer in as few words as possible.
     {context}
     Question: {question}
     """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # type: ignore
        | prompt
        | model
        | StrOutputParser()
    )

    answer = chain.invoke("What city did I visit last?")

    assert "Paris" in answer


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OpenAI for chat responses.",
)
@pytest.mark.asyncio(scope="function")
@pytest.mark.parametrize(
    "async_collection",
    [
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ],
    indirect=True
)
async def test_chain_async(
    async_collection: AsyncCollections,
    embedding: Embeddings,
) -> None:
    """Demonstrate usage of MongoDBAtlasVectorSearch in a realistic chain with async clients.

    Follows example in the docs: https://python.langchain.com/docs/how_to/hybrid/

    Requires OpenAI_API_KEY for embedding and chat model.
    Requires INDEX_NAME to have been set up on MONGODB_URI
    """
    await async_collection.delete_many({})

    search_indexes = [ix async for ix in async_collection.list_search_indexes()]
    if all([INDEX_NAME != ix["name"] for ix in search_indexes]):
        await index.acreate_vector_search_index(
            collection=async_collection,
            index_name=INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            similarity="cosine",
            filters=None,
            wait_until_complete=TIMEOUT,
        )

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=async_collection,
        embedding=embedding,
        index_name=INDEX_NAME,
        text_key="page_content",
    )

    texts = [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
        "In 2019, I visited San Francisco",
        "In 2020, I visited Vancouver",
    ]
    await vectorstore.aadd_texts(texts)

    query = "In the United States, what city did I visit last?"
    k = len(texts)

    store_output = await vectorstore.asimilarity_search(query=query, k=k)
    assert len(store_output) == k
    assert isinstance(store_output[0], Document)

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=k))
    retriever_output = await retriever.ainvoke(query)
    assert len(retriever_output) == len(texts)
    assert retriever_output[0].page_content == store_output[0].page_content

    template = """Answer the question based only on the following context.
     Answer in as few words as possible.
     {context}
     Question: {question}
     """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # type: ignore
        | prompt
        | model
        | StrOutputParser()
    )

    answer = await chain.ainvoke("What city did I visit last?")
    assert "Paris" in answer

    # Cleanup
    await index.adrop_vector_search_index(async_collection, INDEX_NAME, wait_until_complete=TIMEOUT)
