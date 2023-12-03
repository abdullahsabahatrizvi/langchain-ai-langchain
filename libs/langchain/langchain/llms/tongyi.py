from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, cast

from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(llm: Tongyi) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _generate_with_retry(**_kwargs: Any) -> Any:
        resp = llm.client.call(**_kwargs)
        if resp.status_code == 200:
            return resp
        elif resp.status_code in [400, 401]:
            raise ValueError(
                f"status_code: {resp.status_code} \n "
                f"code: {resp.code} \n message: {resp.message}"
            )
        else:
            raise HTTPError(
                f"HTTP error occurred: status_code: {resp.status_code} \n "
                f"code: {resp.code} \n message: {resp.message}",
                response=resp,
            )

    return _generate_with_retry(**kwargs)


def stream_generate_with_retry(llm: Tongyi, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _stream_generate_with_retry(**_kwargs: Any) -> Any:
        stream_resps = []
        resps = llm.client.call(**_kwargs)
        for resp in resps:
            if resp.status_code == 200:
                stream_resps.append(resp)
            elif resp.status_code in [400, 401]:
                raise ValueError(
                    f"status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}"
                )
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}",
                    response=resp,
                )
        return stream_resps

    return _stream_generate_with_retry(**kwargs)


class Tongyi(LLM):
    """Tongyi Qwen large language models.

    To use, you should have the ``dashscope`` python package installed, and the
    environment variable ``DASHSCOPE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import Tongyi
            Tongyi = tongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = "qwen-plus"

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[SecretStr] = None
    """Dashscope api key provide by alicloud."""

    n: int = 1
    """How many completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["dashscope_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        )
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "top_p": self.top_p,
            "api_key": cast(SecretStr, self.dashscope_api_key).get_secret_value(),
        }

        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Tongyi's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = tongyi("Tell me a joke.")
        """
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }

        completion = generate_with_retry(
            self,
            prompt=prompt,
            **params,
        )
        return completion["output"]["text"]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        params: Dict[str, Any] = {
            **{"model": self.model_name},
            **self._default_params,
            **kwargs,
        }
        if self.streaming:
            if len(prompts) > 1:
                raise ValueError("Cannot stream results with multiple prompts.")
            params["stream"] = True
            temp = ""
            for stream_resp in stream_generate_with_retry(
                self, prompt=prompts[0], **params
            ):
                if run_manager:
                    stream_resp_text = stream_resp["output"]["text"]
                    stream_resp_text = stream_resp_text.replace(temp, "")
                    # Ali Cloud's streaming transmission interface, each return content
                    # will contain the output
                    # of the previous round(as of September 20, 2023, future updates to
                    # the Alibaba Cloud API may vary)
                    run_manager.on_llm_new_token(stream_resp_text)
                    # The implementation of streaming transmission primarily relies on
                    # the "on_llm_new_token" method
                    # of the streaming callback.
                temp = stream_resp["output"]["text"]

                generations.append(
                    [
                        Generation(
                            text=stream_resp["output"]["text"],
                            generation_info=dict(
                                finish_reason=stream_resp["output"]["finish_reason"],
                            ),
                        )
                    ]
                )
            generations.reverse()
            # In the official implementation of the OpenAI API,
            # the "generations" parameter passed to LLMResult seems to be a 1*1*1
            # two-dimensional list
            # (including in non-streaming mode).
            # Considering that Alibaba Cloud's streaming transmission
            # (as of September 20, 2023, future updates to the Alibaba Cloud API may
            # vary)
            # includes the output of the previous round in each return,
            # reversing this "generations" list should suffice
            # (This is the solution with the least amount of changes to the source code,
            # while still allowing for convenient modifications in the future,
            # although it may result in slightly more memory consumption).
        else:
            for prompt in prompts:
                completion = generate_with_retry(
                    self,
                    prompt=prompt,
                    **params,
                )
                generations.append(
                    [
                        Generation(
                            text=completion["output"]["text"],
                            generation_info=dict(
                                finish_reason=completion["output"]["finish_reason"],
                            ),
                        )
                    ]
                )
        return LLMResult(generations=generations)
