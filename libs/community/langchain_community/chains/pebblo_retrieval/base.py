"""
Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering
against a vector database.
"""

import datetime
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_community.chains.pebblo_retrieval.enforcement_filters import (
    SUPPORTED_VECTORSTORES,
    set_enforcement_filters,
)
from langchain_community.chains.pebblo_retrieval.models import (
    App,
    AuthContext,
    ChainInfo,
    SemanticContext,
)
from langchain_community.chains.pebblo_retrieval.utilities import (
    CLASSIFIER_URL,
    PLUGIN_VERSION,
    PROMPT_GOV_URL,
    PebbloAPIWrapper,
    get_runtime,
)

logger = logging.getLogger(__name__)


class PebbloRetrievalQA(Chain):
    """
    Retrieval Chain with Identity & Semantic Enforcement for question-answering
    against a vector database.
    """

    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_source_documents: bool = False
    """Return the source documents or not."""

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStore to use for retrieval."""
    auth_context_key: str = "auth_context"  #: :meta private:
    """Authentication context for identity enforcement."""
    semantic_context_key: str = "semantic_context"  #: :meta private:
    """Semantic context for semantic enforcement."""
    app_name: str  #: :meta private:
    """App name."""
    owner: str  #: :meta private:
    """Owner of app."""
    description: str  #: :meta private:
    """Description of app."""
    api_key: Optional[str] = None  #: :meta private:
    """Pebblo cloud API key for app."""
    classifier_url: str = CLASSIFIER_URL  #: :meta private:
    """Classifier endpoint."""
    classifier_location: str = "local"  #: :meta private:
    """Classifier location. It could be either of 'local' or 'pebblo-cloud'."""
    _discover_sent: bool = False  #: :meta private:
    """Flag to check if discover payload has been sent."""
    enable_prompt_gov: bool = True  #: :meta private:
    """Flag to check if prompt governance is enabled or not"""
    pb_client: PebbloAPIWrapper = Field(default_factory=PebbloAPIWrapper)
    """Pebblo API client"""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        prompt_time = datetime.datetime.now().isoformat()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        auth_context = inputs.get(self.auth_context_key)
        semantic_context = inputs.get(self.semantic_context_key)
        _, prompt_entities = self._check_prompt_validity(question)

        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(
                question, auth_context, semantic_context, run_manager=_run_manager
            )
        else:
            docs = self._get_docs(question, auth_context, semantic_context)  # type: ignore[call-arg]
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        self.pb_client.send_prompt(
            self.app_name,
            self.retriever,
            question,
            answer,
            auth_context,
            docs,
            prompt_entities,
            prompt_time,
            self.enable_prompt_gov,
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        auth_context = inputs.get(self.auth_context_key)
        semantic_context = inputs.get(self.semantic_context_key)
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )

        _, prompt_entities = self._check_prompt_validity(question)

        if accepts_run_manager:
            docs = await self._aget_docs(
                question, auth_context, semantic_context, run_manager=_run_manager
            )
        else:
            docs = await self._aget_docs(question, auth_context, semantic_context)  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        extra = "forbid"

    @property
    def input_keys(self) -> List[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key, self.auth_context_key, self.semantic_context_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys += ["source_documents"]
        return _output_keys

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "pebblo_retrieval_qa"

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        app_name: str,
        description: str,
        owner: str,
        chain_type: str = "stuff",
        chain_type_kwargs: Optional[dict] = None,
        api_key: Optional[str] = None,
        classifier_url: str = CLASSIFIER_URL,
        classifier_location: str = "local",
        **kwargs: Any,
    ) -> "PebbloRetrievalQA":
        """Load chain from chain type."""
        from langchain.chains.question_answering import load_qa_chain

        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm, chain_type=chain_type, **_chain_type_kwargs
        )

        # generate app
        app: App = PebbloRetrievalQA._get_app_details(
            app_name=app_name,
            description=description,
            owner=owner,
            llm=llm,
            **kwargs,
        )
        # initialize Pebblo API client
        pb_client = PebbloAPIWrapper(
            api_key=api_key,
            classifier_location=classifier_location,
            classifier_url=classifier_url,
        )
        # send app discovery request
        pb_client.send_app_discover(app)
        return cls(
            combine_documents_chain=combine_documents_chain,
            app_name=app_name,
            owner=owner,
            description=description,
            api_key=api_key,
            classifier_url=classifier_url,
            classifier_location=classifier_location,
            pb_client=pb_client,
            **kwargs,
        )

    @validator("retriever", pre=True, always=True)
    def validate_vectorstore(
        cls, retriever: VectorStoreRetriever
    ) -> VectorStoreRetriever:
        """
        Validate that the vectorstore of the retriever is supported vectorstores.
        """
        if retriever.vectorstore.__class__.__name__ not in SUPPORTED_VECTORSTORES:
            raise ValueError(
                f"Vectorstore must be an instance of one of the supported "
                f"vectorstores: {SUPPORTED_VECTORSTORES}. "
                f"Got '{retriever.vectorstore.__class__.__name__}' instead."
            )
        return retriever

    def _get_docs(
        self,
        question: str,
        auth_context: Optional[AuthContext],
        semantic_context: Optional[SemanticContext],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        set_enforcement_filters(self.retriever, auth_context, semantic_context)
        return self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

    async def _aget_docs(
        self,
        question: str,
        auth_context: Optional[AuthContext],
        semantic_context: Optional[SemanticContext],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        set_enforcement_filters(self.retriever, auth_context, semantic_context)
        return await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

    @staticmethod
    def _get_app_details(  # type: ignore
        app_name: str, owner: str, description: str, llm: BaseLanguageModel, **kwargs
    ) -> App:
        """Fetch app details. Internal method.
        Returns:
            App: App details.
        """
        framework, runtime = get_runtime()
        chains = PebbloRetrievalQA.get_chain_details(llm, **kwargs)
        app = App(
            name=app_name,
            owner=owner,
            description=description,
            runtime=runtime,
            framework=framework,
            chains=chains,
            plugin_version=PLUGIN_VERSION,
        )
        return app

    @classmethod
    def set_discover_sent(cls) -> None:
        cls._discover_sent = True

    def _check_prompt_validity(self, question: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the validity of the given prompt using a remote classification service.

        This method sends a prompt to a remote classifier service and return entities
        present in prompt or not.

        Args:
            question (str): The prompt question to be validated.

        Returns:
            bool: True if the prompt is valid (does not contain deny list entities),
            False otherwise.
            dict: The entities present in the prompt
        """

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        prompt_payload = {"prompt": question}
        is_valid_prompt: bool = True
        prompt_gov_api_url = f"{self.classifier_url}{PROMPT_GOV_URL}"
        pebblo_resp = None
        prompt_entities: dict = {"entities": {}, "entityCount": 0}
        if self.classifier_location == "local":
            try:
                pebblo_resp = requests.post(
                    prompt_gov_api_url,
                    headers=headers,
                    json=prompt_payload,
                    timeout=20,
                )

                logger.debug("prompt-payload: %s", prompt_payload)
                logger.debug(
                    "send_prompt[local]: request url %s, body %s len %s\
                        response status %s body %s",
                    pebblo_resp.request.url,
                    str(pebblo_resp.request.body),
                    str(
                        len(
                            pebblo_resp.request.body if pebblo_resp.request.body else []
                        )
                    ),
                    str(pebblo_resp.status_code),
                    pebblo_resp.json(),
                )
                logger.debug(f"pebblo_resp.json() {pebblo_resp.json()}")
                prompt_entities["entities"] = pebblo_resp.json().get("entities", {})
                prompt_entities["entityCount"] = pebblo_resp.json().get(
                    "entityCount", 0
                )

            except requests.exceptions.RequestException:
                logger.warning("Unable to reach pebblo server.")
            except Exception as e:
                logger.warning("An Exception caught in _send_discover: local %s", e)
        return is_valid_prompt, prompt_entities

    @classmethod
    def get_chain_details(cls, llm: BaseLanguageModel, **kwargs) -> List[ChainInfo]:
        """
        Get chain details.

        Args:
            llm (BaseLanguageModel): Language model instance.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Dict[str, Any]]: Chain details.
        """
        llm_dict = llm.__dict__
        chains = [
            {
                "name": cls.__name__,
                "model": {
                    "name": llm_dict.get("model_name", llm_dict.get("model")),
                    "vendor": llm.__class__.__name__,
                },
                "vector_dbs": [
                    {
                        "name": kwargs["retriever"].vectorstore.__class__.__name__,
                        "embedding_model": str(
                            kwargs["retriever"].vectorstore._embeddings.model
                        )
                        if hasattr(kwargs["retriever"].vectorstore, "_embeddings")
                        else (
                            str(kwargs["retriever"].vectorstore._embedding.model)
                            if hasattr(kwargs["retriever"].vectorstore, "_embedding")
                            else None
                        ),
                    }
                ],
            },
        ]
        chains = [ChainInfo(**chain) for chain in chains]
        return chains
