import pytest
from uuid import uuid4
import duckdb
import os
from langchain_community.vectorstores import DuckDB
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

@pytest.fixture
def duckdb_connection():
    import duckdb
    # Setup a temporary DuckDB database
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()

@pytest.fixture
def embeddings():
    return FakeEmbeddings()

@pytest.fixture
def texts():
    return ["text 1", "text 2", "item 3"]

@pytest.mark.requires("duckdb")
def test_duckdb_with_connection(duckdb_connection, embeddings, texts):
    store = DuckDB(connection=duckdb_connection, embedding=embeddings, table_name="test_table")
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

@pytest.mark.requires("duckdb")
def test_duckdb_without_connection(embeddings, texts):
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(texts)
    result = store.similarity_search("text 1")
    result_texts = [doc.page_content for doc in result]
    assert "text 1" in result_texts

@pytest.mark.requires("duckdb")
def test_duckdb_add_texts(embeddings):
    store = DuckDB(embedding=embeddings, table_name="test_table")
    store.add_texts(["text 2"])
    result = store.similarity_search("text 2")
    result_texts = [doc.page_content for doc in result]
    assert "text 2" in result_texts

@pytest.mark.requires("duckdb")
def test_duckdb_add_texts_with_metadata(duckdb_connection, embeddings):
    store = DuckDB(connection=duckdb_connection, embedding=embeddings, table_name="test_table_with_metadata")
    texts = ["text with metadata 1", "text with metadata 2"]
    metadatas = [
        {"author": "Author 1", "date": "2021-01-01"},
        {"author": "Author 2", "date": "2021-02-01"}
    ]
    
    # Add texts along with their metadata
    store.add_texts(texts, metadatas=metadatas)
    
    # Perform a similarity search to retrieve the documents
    result = store.similarity_search("text with metadata", k=2)
    
    # Check if the metadata is correctly associated with the texts
    assert len(result) == 2, "Should return two results"
    assert result[0].metadata.get("author") == "Author 1", "Metadata for Author 1 should be correctly retrieved"
    assert result[0].metadata.get("date") == "2021-01-01", "Date for Author 1 should be correctly retrieved"
    assert result[1].metadata.get("author") == "Author 2", "Metadata for Author 2 should be correctly retrieved"
    assert result[1].metadata.get("date") == "2021-02-01", "Date for Author 2 should be correctly retrieved"

@pytest.mark.requires("duckdb")
def test_duckdb_add_texts_with_predefined_ids(duckdb_connection, embeddings):
    store = DuckDB(connection=duckdb_connection, embedding=embeddings, table_name="test_table_predefined_ids")
    texts = ["unique text 1", "unique text 2"]
    predefined_ids = [str(uuid4()), str(uuid4())]  # Generate unique IDs
    
    # Add texts with the predefined IDs
    store.add_texts(texts, ids=predefined_ids)
    
    # Perform a similarity search for each text and check if it's found
    for text in texts:
        result = store.similarity_search(text)
        
        found_texts = [doc.page_content for doc in result]
        assert text in found_texts, f"Text '{text}' was not found in the search results."