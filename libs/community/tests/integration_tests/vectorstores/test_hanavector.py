"""Test HANA vectorstore functionality."""
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import ConsistentFakeEmbeddings
from hdbcli import dbapi
import os


try:
    from hdbcli import dbapi
    hanadb_installed = True
except ImportError:
    hanadb_installed = False

embedding = ConsistentFakeEmbeddings()
connection = dbapi.connect(
        address=os.environ.get("DB_ADDRESS"),
        port=30015,
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        autocommit=False,
        sslValidateCertificate=False
)

@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]

def drop_table(connection, table_name):
    try:
        cur = connection.cursor()
        sql_str = f"DROP TABLE {table_name}"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_add_texts(texts: List[str]) -> None:
    """Test end to end construction and search."""
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name="WTF")
    vectordb.add_texts(texts)
    results = vectordb.similarity_search("foo", k=1)
    assert results == [Document(page_content="foo")]

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table(texts: List[str]) -> None:
    """Test end to end construction and search."""
    table_name = "NON_EXISTING"
    # Delete table if it exists
    drop_table(connection, table_name)

    # Check if table is created
    vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)

    assert vectordb._table_exists(table_name)

@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_missing_columns(texts: List[str]) -> None:
    table_name = "EXISTING_WRONG"
    try:
        drop_table(connection, table_name)
        cur = connection.cursor()
        sql_str = f"CREATE TABLE {table_name}(WRONG_COL NVARCHAR(500));"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)
        exception_occured = False
    except:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_wrong_typed_columns(texts: List[str]) -> None:
    table_name = "EXISTING_WRONG"
    content_field = "DOC_TEXT"
    metadata_field = "DOC_META"
    vector_field = "DOC_VECTOR"
    try:
        drop_table(connection, table_name)
        cur = connection.cursor()
        sql_str = f"CREATE TABLE {table_name}({content_field} INTEGER, {metadata_field} INTEGER, {vector_field} INTEGER);"
        cur.execute(sql_str)
    except:
        pass        
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        vectordb = HanaDB(connection=connection, embedding=embedding, distance_strategy = DistanceStrategy.COSINE, table_name=table_name)
        exception_occured = False
    except AttributeError as err:
        print(err)
        exception_occured = True
    assert exception_occured
