import os
from typing import List, Generator, AsyncGenerator

import pytest
import pytest_asyncio
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo
from motor.motor_asyncio import AsyncIOMotorClient

# For the beta async client from pymongo
from pymongo import AsyncMongoClient

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

@pytest.fixture(scope="session")
def technical_report_pages() -> List[Document]:
    """Returns a Document for each of the 100 pages of a GPT-4 Technical Report"""
    loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
    pages = loader.load()
    return pages


@pytest.fixture(scope="session")
def connection_string() -> str:
    return MONGODB_URI


@pytest.fixture(scope="session")
def client() -> Generator[MongoClient, None, None]:
    """Sync client fixture."""
    client = MongoClient(
        MONGODB_URI,
        driver=DriverInfo(name="Langchain Tests", version="test-version")
    )
    yield client
    client.close()

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the event loop for the entire test session.
    By default, pytest-asyncio uses function-level loops, 
    which can sometimes cause unexpected closures across modules.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def motor_client() -> AsyncGenerator[AsyncIOMotorClient, None]:
    """Motor async client fixture."""
    client = AsyncIOMotorClient(
        MONGODB_URI,
        driver=DriverInfo(name="Langchain Tests", version="test-version")
    )
    yield client
    client.close()

@pytest_asyncio.fixture(scope="session")
async def pymongo_async_client() -> AsyncGenerator[AsyncMongoClient, None]:
    """PyMongo beta async client fixture."""
    client = AsyncMongoClient(
        MONGODB_URI,
        driver=DriverInfo(name="Langchain Tests", version="test-version")
    )
    yield client
    client.close()

@pytest.fixture(scope="session")
def embedding() -> Embeddings:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
            model="text-embedding-3-small",
        )

    return OllamaEmbeddings(model="all-minilm:l6-v2")


@pytest.fixture(scope="session")
def dimensions() -> int:
    if os.environ.get("OPENAI_API_KEY"):
        return 1536
    return 384
