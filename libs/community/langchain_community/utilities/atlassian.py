"""Util that calls Atlassian APIs."""

import json
import re
import inspect
from typing import Any, Optional, List

from pydantic import BaseModel, ConfigDict
from langchain_core.utils import get_from_dict_or_env


class AtlassianAPIWrapper(BaseModel):
    """Wrapper for Atlassian APIs, incluindo Jira e Confluence."""

    jira: Optional[Any] = None
    confluence: Optional[Any] = None
    atlassian_username: Optional[str] = None
    atlassian_api_token: Optional[str] = None
    atlassian_instance_url: Optional[str] = None
    atlassian_cloud: Optional[bool] = None
    filter_keys: Optional[List[str]] = None

    model_config = ConfigDict(extra='forbid')

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_and_initialize_environment()

    def validate_and_initialize_environment(self):
        """Valida se a chave API e o pacote Python estão presentes no ambiente
        e inicializa as instâncias do Jira e Confluence."""

        def get_env_var(var_name: str, env_name: str, default: Optional[str] = None) -> str:
            return get_from_dict_or_env(self.__dict__, var_name, env_name, default)

        self.atlassian_username = get_env_var("atlassian_username", "ATLASSIAN_USERNAME", "")
        self.atlassian_api_token = get_env_var("atlassian_api_token", "ATLASSIAN_API_TOKEN")
        self.atlassian_instance_url = get_env_var("atlassian_instance_url", "ATLASSIAN_INSTANCE_URL")
        self.atlassian_cloud = bool(get_env_var("atlassian_cloud", "ATLASSIAN_CLOUD"))

        try:
            from atlassian import Confluence, Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api não está instalado. "
                "Por favor, instale com `pip install atlassian-python-api`"
            )

        if self.atlassian_username == "":
            self.jira = Jira(
                url=self.atlassian_instance_url,
                token=self.atlassian_api_token,
                cloud=self.atlassian_cloud,
            )
        else:
            self.jira = Jira(
                url=self.atlassian_instance_url,
                username=self.atlassian_username,
                password=self.atlassian_api_token,
                cloud=self.atlassian_cloud,
            )

        self.confluence = Confluence(
            url=self.atlassian_instance_url,
            username=self.atlassian_username,
            password=self.atlassian_api_token,
            cloud=self.atlassian_cloud,
        )

        # Verificar se as instâncias Jira e Confluence são inicializadas corretamente
        assert self.jira is not None, "Failed to initialize Jira instance."
        assert self.confluence is not None, "Failed to initialize Confluence instance."

        print(f"Jira instance: {self.jira}")
        print(f"Confluence instance: {self.confluence}")

        if not self.jira or not self.confluence:
            raise ValueError("Failed to initialize Jira or Confluence instances.")

    def filter_response_keys(self, response: dict) -> dict:
        """Remove specified keys from the response dictionary using wildcard patterns."""
        if not self.filter_keys:
            return response

        # Remove leading/trailing spaces and transform wildcard patterns into regex patterns
        filter_patterns = [re.compile('^' + key.strip().replace('*', '.*') + '$') for key in self.filter_keys]
        print(f"Filter patterns: {filter_patterns}")  # Adicionado para debug

        def recursive_filter(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: recursive_filter(v) for k, v in data.items() if not any(p.match(k) for p in filter_patterns)}
            elif isinstance(data, list):
                return [recursive_filter(item) for item in data]
            else:
                return data

        return recursive_filter(response)

    def run(self, mode: str, query: str) -> str:
        try:
            if mode == "jira_jql":
                return self.jira_jql(query)
            elif mode == "jira_get_functions":
                return self.get_jira_functions(query)
            elif mode == "jira_other":
                return self.jira_other(query)
            elif mode == "confluence_cql":
                return self.confluence_cql(query)
            elif mode == "confluence_get_functions":
                return self.get_confluence_functions(query)
            elif mode == "confluence_other":
                return self.confluence_other(query)
            else:
                raise ValueError(f"Got unexpected mode {mode}")
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    def get_jira_functions(self, query: Optional[str] = None) -> str:
        all_attributes = dir(self.jira)
        functions = [attr for attr in all_attributes if
                     callable(getattr(self.jira, attr)) and not attr.startswith('_')]

        if query is None or query == "":
            return json.dumps({"functions": functions}, indent=2)
        elif query in functions:
            function_obj = getattr(self.jira, query)
            params = inspect.signature(function_obj).parameters
            return json.dumps({"function": query, "parameters": list(params.keys())}, indent=2)
        else:
            return json.dumps({"error": f"Function {query} not found."}, indent=2)

    def get_confluence_functions(self, query: Optional[str] = None) -> str:
        all_attributes = dir(self.confluence)
        functions = [attr for attr in all_attributes if
                     callable(getattr(self.confluence, attr)) and not attr.startswith('_')]

        if query is None or query == "":
            return json.dumps({"functions": functions}, indent=2)
        elif query in functions:
            function_obj = getattr(self.confluence, query)
            params = inspect.signature(function_obj).parameters
            return json.dumps({"function": query, "parameters": list(params.keys())}, indent=2)
        else:
            return json.dumps({"error": f"Function {query} not found."}, indent=2)

    def jira_other(self, query: str) -> str:
        try:
            params = json.loads(query)
            function_name = params.get("function")

            accepted_params = json.loads(self.get_jira_functions(function_name))
            if "error" in accepted_params:
                raise ValueError(accepted_params["error"])

            jira_function = getattr(self.jira, function_name)

            args = params.get("args", [])
            kwargs = params.get("kwargs", {})

            presented_params = args + list(kwargs.keys())

            if not set(presented_params).issubset(set(accepted_params["parameters"])):
                raise ValueError(
                    f"Function '{function_name}' accepts {accepted_params['parameters']} parameters. But got: {presented_params}."
                )

            result = jira_function(*args, **kwargs)
            if self.filter_keys:
                return self.apply_filter(result)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    def confluence_other(self, query: str) -> str:
        try:
            params = json.loads(query)
            function_name = params.get("function")

            accepted_params = json.loads(self.get_confluence_functions(function_name))
            if "error" in accepted_params:
                raise ValueError(accepted_params["error"])

            confluence_function = getattr(self.confluence, function_name)

            args = params.get("args", [])
            kwargs = params.get("kwargs", {})

            presented_params = args + list(kwargs.keys())

            if not set(presented_params).issubset(set(accepted_params["parameters"])):
                raise ValueError(
                    f"Function '{function_name}' accepts {accepted_params['parameters']} parameters. But got: {presented_params}."
                )

            result = confluence_function(*args, **kwargs)
            if self.filter_keys:
                return self.apply_filter(result)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    def jira_jql(self, query: str) -> str:
        issues = self.jira.jql(query)
        if self.filter_keys:
            return self.apply_filter(issues)
        return json.dumps(issues, indent=2)

    def confluence_cql(self, query: str) -> str:
        content = self.confluence.cql(query)
        if self.filter_keys:
            return self.apply_filter(content)
        return json.dumps(content, indent=2)

    def apply_filter(self, result: Any) -> str:
        """Apply key filter to the result."""
        if isinstance(result, dict):
            filtered_result = self.filter_response_keys(result)
            return json.dumps(filtered_result, indent=2)
        else:
            return json.dumps(result, indent=2)
