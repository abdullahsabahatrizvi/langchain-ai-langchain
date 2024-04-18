# -----------------------------------------------------------------------------
# Copyright (c) 2023 - , Oracle and/or its affiliates.
# -----------------------------------------------------------------------------
# Authors:
#   Harichandan Roy (hroy)
#   David Jiang (ddjiang)
#
# -----------------------------------------------------------------------------
# oracleai.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, List, Optional

import oracledb
from langchain_core.documents import Document
from oracledb import Connection

logger = logging.getLogger(__name__)

"""OracleSummary class"""


class OracleSummary:
    """Get Summary
    Args:
        conn: Oracle Connection,
        params: Summary parameters,
        proxy: Proxy
    """

    def __init__(
        self, conn: Connection, params: Dict[str, Any], proxy: Optional[str] = None
    ):
        self.conn = conn
        self.proxy = proxy
        self.summary_params = params

    def get_summary(self, docs) -> List[str]:
        """Get the summary of the input docs.
        Args:
            docs: The documents to generate summary for.
                  Allowed input types: str, Document, List[str], List[Document]
        Returns:
            List of summary text, one for each input doc.
        """

        if docs is None:
            return None

        results = []
        try:
            oracledb.defaults.fetch_lobs = False
            cursor = self.conn.cursor()

            if self.proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self.proxy
                )

            if isinstance(docs, str):
                results = []

                summary = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.execute(
                    """
                    declare
                        input clob;
                    begin
                        input := :data;
                        :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                    end;""",
                    data=docs,
                    params=json.dumps(self.summary_params),
                    summ=summary,
                )

                if summary is None:
                    results.append("")
                else:
                    results.append(str(summary.getvalue()))

            elif isinstance(docs, Document):
                results = []

                summary = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.execute(
                    """
                    declare
                        input clob;
                    begin
                        input := :data;
                        :summ := dbms_vector_chain.utl_to_summary(input, json(:params));
                    end;""",
                    data=docs.page_content,
                    params=json.dumps(self.summary_params),
                    summ=summary,
                )

                if summary is None:
                    results.append("")
                else:
                    results.append(str(summary.getvalue()))

            elif isinstance(docs, List):
                results = []

                for doc in docs:
                    summary = cursor.var(oracledb.DB_TYPE_CLOB)
                    if isinstance(doc, str):
                        cursor.execute(
                            """
                            declare
                                input clob;
                            begin
                                input := :data;
                                :summ := dbms_vector_chain.utl_to_summary(input, 
                                            json(:params));
                            end;""",
                            data=doc,
                            params=json.dumps(self.summary_params),
                            summ=summary,
                        )

                    elif isinstance(doc, Document):
                        cursor.execute(
                            """
                            declare
                                input clob;
                            begin
                                input := :data;
                                :summ := dbms_vector_chain.utl_to_summary(input, 
                                            json(:params));
                            end;""",
                            data=doc.page_content,
                            params=json.dumps(self.summary_params),
                            summ=summary,
                        )

                    else:
                        raise Exception("Invalid input type")

                    if summary is None:
                        results.append("")
                    else:
                        results.append(str(summary.getvalue()))

            else:
                raise Exception("Invalid input type")

            cursor.close()
            return results

        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            cursor.close()
            raise


# uncomment the following code block to run the test

"""
# A sample unit test.

''' get the Oracle connection '''
conn = oracledb.connect(
    user="ut",
    password="ut",
    dsn="phoenix104636.dev3sub3phx.databasede3phx.oraclevcn.com/v5.regress.rdbms.dev.us.oracle.com")
print("Oracle connection is established...")

''' params '''
summary_params = {"provider": "database","glevel": "S",
                  "numParagraphs": 1,"language": "english"} 
proxy = "www-proxy-ash7.us.oracle.com:80"

''' instance '''
summ = OracleSummary(conn=conn, params=summary_params, proxy=proxy)

summary = summ.get_summary("In the heart of the forest, " + 
    "a lone fox ventured out at dusk, seeking a lost treasure. " + 
    "With each step, memories flooded back, guiding its path. " + 
    "As the moon rose high, illuminating the night, the fox unearthed " + 
    "not gold, but a forgotten friendship, worth more than any riches.")
print(f"Summary generated by OracleSummary: {summary}")

conn.close()
print("Connection is closed.")

"""
