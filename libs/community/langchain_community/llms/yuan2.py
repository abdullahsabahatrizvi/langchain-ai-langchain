import logging
from typing import Any, List, Mapping, Optional, Set, Dict

import requests
import json
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class Yuan2(LLM):
    """Yuan2.0 language models.

    Example:
        .. code-block:: python

            yuan_llm = Yuan2(infer_api="http://127.0.0.1:8000/yuan", max_tokens=1024, temp=1.0, top_p=0.9, top_k=40)
            print(yuan_llm)
            print(yuan_llm("你是谁？"))
    """
    infer_api: str = "http://127.0.0.1:8000/yuan"
    """Yuan2.0 inference api"""

    max_tokens: int = Field(1024, alias="max_token")
    """Token context window."""

    temp: Optional[float] = 0.7
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.9
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    do_sample: bool = False
    """The do_sample is a Boolean value that determines whether to use the sampling method during text generation."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_last_n: Optional[int] = 64
    "Last n tokens to penalize"

    repeat_penalty: Optional[float] = 1.18
    """The penalty to apply to repeated tokens."""

    streaming: bool = False
    """Whether to stream the results or not."""

    history: List[str] = []
    """History of the conversation"""

    use_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "Yuan2.0"

    @staticmethod
    def _model_param_names() -> Set[str]:
        return {
            "max_tokens",
            "temp",
            "top_k",
            "top_p",
            "do_sample",
        }

    def _default_params(self) -> Dict[str, Any]:
        return {
            "infer_api": self.infer_api,
            "max_tokens": self.max_tokens,
            "temp": self.temp,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "use_history": self.use_history
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self._llm_type,
            **self._default_params(),
            **{
                k: v for k, v in self.__dict__.items() if k in self._model_param_names()
            },
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to a Yuan2.0 LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = yuan_llm("你能做什么?")
        """

        if self.use_history:
            self.history.append(prompt)
            input = "\n".join(self.history)
        else:
            input = prompt

        headers = {
            'Content-Type': 'application/json'
        }
        data = json.dumps({
            "ques_list":[
                {
                    "id": "000",
                    "ques": input
                }
            ],
            "tokens_to_generate": self.max_tokens,
            "temperature": self.temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
        })

        logger.debug("Yuan2.0 prompt:", input)

        # call api
        try:
            response = requests.put(self.infer_api, headers=headers, data=data)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference api: {e}")

        logger.debug(f"Yuan2.0 response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")
        try:
            resp = response.json()

            if resp["errCode"] != "0":
                raise ValueError(f"Failed with error code [{resp['errCode']}], error message: [{resp['errMessage']}]")

            if "resData" in resp:
                if len(resp["resData"]["output"]) >= 0:
                    generate_text = resp["resData"]["output"][0]["ans"]
                else:
                    raise ValueError("No output found in response.")
            else:
                raise ValueError("No resData found in response.")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference api: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            generate_text = enforce_stop_tokens(generate_text, stop)

        # support multi-turn chat
        if self.use_history:
            self.history.append(generate_text)

        logger.debug(f"history: {self.history}")
        return generate_text

