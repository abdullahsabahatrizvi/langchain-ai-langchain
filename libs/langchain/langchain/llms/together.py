"""Wrapper around Together's Generation API."""
import logging
from typing import Any, Dict, List, Optional

from aiohttp import ClientSession

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.utilities.requests import Requests
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Together(LLM):
    """Wrapper around together models.

    To use, you should have
    the environment variable ``TOGETHER_API_KEY`` set with your API token.
    You can find your token here: https://api.together.xyz/settings/api-keys


    params format {
        "model": "togethercomputer/RedPajama-INCITE-7B-Instruct",
        "prompt": "Q: The capital of France is?\nA:",
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "max_tokens": 1,
        "repetition_penalty": 1
    }

    for api reference check together documentation:
    https://docs.together.ai/docs/inference-rest
    """

    base_url: str = "https://api.together.xyz/inference"

    together_api_key: Optional[str] = None

    model: Optional[str] = None
    """
    model name for above provider (eg: 'togethercomputer/RedPajama-INCITE-Chat-3B-v1' 
    for RedPajama-INCITE Chat (3B))
    available models are shown on https://docs.together.ai/docs/inference-models 
    """

    temperature: Optional[float] = Field(default=0.7, ge=0, le=1)  # for text
    top_p: Optional[float] = Field(default=0.7, ge=0, le=1)  # for text
    top_k: Optional[int] = Field(default=50, ge=0)  # for text
    max_tokens: Optional[int] = Field(default=128, ge=0)  # for text
    repetition_penalty: Optional[float] = Field(default=1, ge=0, le=1)  # for text

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["together_api_key"] = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "together"

    def _format_output(self, output: dict) -> str:
        return output["output"]["choices"][0]["text"]

    @staticmethod
    def get_user_agent() -> str:
        from langchain import __version__

        return f"langchain/{__version__}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Together's text generation endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            json formatted str response.
        """

        url = f"{self.base_url}"
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

        # Overwrite the values in payload with kwargs if they are provided
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "repetition_penalty",
        ]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # filter None values to not pass them to the http payload
        payload = {k: v for k, v in payload.items() if v is not None}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)

        if response.status_code >= 500:
            raise Exception(f"Together Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"Together received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"Together returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()

        if data.get("status") != "finished":
            err_msg = data.get("error", "Undefined Error")
            raise Exception(err_msg)

        output = self._format_output(data)

        return output

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Together model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.
        """

        stops = None
        if self.stop_sequences is not None and stop is not None:
            raise ValueError(
                "stop sequences found in both the input and default params."
            )
        elif self.stop_sequences is not None:
            stops = self.stop_sequences
        else:
            stops = stop

        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

        # filter `None` values to not pass them to the http payload as null
        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                if response.status >= 500:
                    raise Exception(f"Together Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"Together received an invalid payload: {response.text}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"Together returned an unexpected response with status "
                        f"{response.status}: {response.text}"
                    )

                response_json = await response.json()

                if response_json.get("status") != "finished":
                    err_msg = response_json.get("error", "Undefined Error")
                    raise Exception(err_msg)

                output = self._format_output(response_json)
                if stops is not None:
                    output = enforce_stop_tokens(output, stops)

                return output
