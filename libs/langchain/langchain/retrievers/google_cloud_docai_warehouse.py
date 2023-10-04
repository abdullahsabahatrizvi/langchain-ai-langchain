"""Retriever wrapper for Google Cloud DocAI Warehouse on Gen App Builder."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.docstore.document import Document
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseRetriever
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from google.cloud.contentwarehouse_v1 import (
        DocumentServiceClient,
        RequestMetadata,
        SearchDocumentsRequest,
    )
    from google.cloud.contentwarehouse_v1.services.document_service.pagers import (
        SearchDocumentsPager,
    )


class GoogleDocaiWarehouseSearchRetriever(BaseRetriever):
    """A retriever based on DocAI Warehouse.

    Documents should be created and documents should be uploaded
        in a separate flow, and this retriever uses only DocAI
        schema_id provided to search for revelant documents.

    More info: https://cloud.google.com/document-ai-warehouse.
    """

    location: str = "us"
    "GCP location where DocAI Warehouse is placed."
    project_id: str
    "GCP project number, should contain digits only."
    schema_id: Optional[str] = None
    "DocAI Warehouse schema to queary against. If nothing is provided, all documents "
    "in the project will be searched."
    qa_size_limit: int = 5
    "The limit on the number of documents returned."
    client: "DocumentServiceClient" = None  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates the environment."""
        try:  # noqa: F401
            from google.cloud.contentwarehouse_v1 import (
                DocumentServiceClient,
            )
        except ImportError as exc:
            raise ImportError(
                "google.cloud.contentwarehouse is not installed."
                "Please install it with pip install google-cloud-contentwarehouse"
            ) from exc

        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")
        values["client"] = DocumentServiceClient()
        return values

    @property
    def _parent(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}"

    @property
    def _schemas(self) -> List[str]:
        if self.schema_id:
            return [f"{self._parent}/documentSchemas/{self.schema_id}"]
        return []

    def _prepare_request_metadata(self, user_ldap: str) -> "RequestMetadata":
        from google.cloud.contentwarehouse_v1 import RequestMetadata, UserInfo

        user_info = UserInfo(id=f"user:{user_ldap}")
        return RequestMetadata(user_info=user_info)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        request = self._prepare_search_request(query, **kwargs)
        response = self.client.search_documents(request=request)
        return self._parse_search_response(response=response)

    def _prepare_search_request(
        self, query: str, **kwargs: Any
    ) -> "SearchDocumentsRequest":
        from google.cloud.contentwarehouse_v1 import (
            DocumentQuery,
            SearchDocumentsRequest,
        )

        try:
            user_ldap = kwargs["user_ldap"]
        except KeyError:
            raise ValueError("Argument user_ldap should be provided!")

        request_metadata = self._prepare_request_metadata(user_ldap=user_ldap)

        return SearchDocumentsRequest(
            parent=self._parent,
            request_metadata=request_metadata,
            document_query=DocumentQuery(
                query=query, is_nl_query=True, document_schema_names=self._schemas
            ),
            qa_size_limit=self.qa_size_limit,
        )

    def _parse_search_response(
        self, response: "SearchDocumentsPager"
    ) -> List[Document]:
        documents = []
        for doc in response.matching_documents:
            metadata = {
                "title": doc.document.title,
                "source": doc.document.raw_document_path,
            }
            documents.append(
                Document(page_content=doc.search_text_snippet, metadata=metadata)
            )
        return documents
