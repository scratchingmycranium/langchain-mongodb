from typing import List
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient, AsyncMongoClient

from langchain_core.documents import Document
from langchain_mongodb.docstores import MongoDBDocStore, AsyncMongoDBDocStore
from ..utils import AsyncCollections

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_docstore"


def test_docstore(client: MongoClient, technical_report_pages: List[Document]) -> None:
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)
    clxn = db[COLLECTION_NAME]

    n_docs = len(technical_report_pages)
    assert clxn.count_documents({}) == 0
    docstore = MongoDBDocStore(collection=clxn)

    docstore.mset([(str(i), technical_report_pages[i]) for i in range(n_docs)])
    assert clxn.count_documents({}) == n_docs

    twenties = list(docstore.yield_keys(prefix="2"))
    assert len(twenties) == 11  # includes 2, 20, 21, ..., 29

    docstore.mdelete([str(i) for i in range(20, 30)] + ["2"])
    assert clxn.count_documents({}) == n_docs - 11
    assert set(docstore.mget(twenties)) == {None}

    sample = docstore.mget(["8", "16", "24", "36"])
    assert sample[2] is None
    assert all(isinstance(sample[i], Document) for i in [0, 1, 3])


@pytest.mark.asyncio(scope="function")
@pytest.mark.parametrize(
    "client_name",
    [
        pytest.param("pymongo", id="pymongo"),
        pytest.param("motor", id="motor")
    ]
)
async def test_docstore_with_async_clients(
    client_name: str,
    pymongo_async_client: AsyncMongoClient,
    motor_client: AsyncIOMotorClient,
    technical_report_pages: List[Document]
) -> None:
    """Test docstore with both async clients (PyMongo and Motor)."""
    client = pymongo_async_client if client_name == "pymongo" else motor_client
    db = client[DB_NAME]
    await db.drop_collection(COLLECTION_NAME)
    clxn = db[COLLECTION_NAME]

    n_docs = len(technical_report_pages)
    assert await clxn.count_documents({}) == 0
    docstore = AsyncMongoDBDocStore(collection=clxn)

    await docstore.amset([(str(i), technical_report_pages[i]) for i in range(n_docs)])
    assert await clxn.count_documents({}) == n_docs

    twenties = [key async for key in docstore.ayield_keys(prefix="2")]
    assert len(twenties) == 11  # includes 2, 20, 21, ..., 29

    await docstore.amdelete([str(i) for i in range(20, 30)] + ["2"])
    assert await clxn.count_documents({}) == n_docs - 11
    assert set(await docstore.amget(twenties)) == {None}

    sample = await docstore.amget(["8", "16", "24", "36"])
    assert sample[2] is None
    assert all(isinstance(sample[i], Document) for i in [0, 1, 3])
