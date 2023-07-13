from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.anthropic import _AnthropicCommon
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    """
    Wrapper around Anthropic's large language model.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            import anthropic
            from langchain.llms import Anthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Returns a dictionary containing the API key for Anthropic's LLM."""
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Returns the type of chat model."""
        return "anthropic-chat"

    @property
    def lc_serializable(self) -> bool:
        """Returns a boolean indicating whether the model is serializable."""
        return True

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        """Converts a single message into a formatted string."""
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"{self.HUMAN_PROMPT} {message.content}"
        elif isinstance(message, AIMessage):
            message_text = f"{self.AI_PROMPT} {message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"{self.HUMAN_PROMPT} <admin>{message.content}</admin>"
        else:
            raise ValueError(f"Got unknown message type: {type(message).__name__}")
        return message_text

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Formats a list of messages into a single string."""
        return "".join(
            self._convert_one_message_to_text(message) for message in messages
        )

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Formats a list of messages into a full prompt for the Anthropic model."""
        messages = messages.copy()  # don't mutate the original list

        if not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return text.rstrip()

    def _generate_response(self, params: Dict[str, Any], run_manager: Optional[Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]] = None) -> str:
        """Generates a response from the Anthropic model based on the provided parameters."""
        if self.streaming:
            completion = ""
            stream_resp = self.client.completions.create(**params, stream=True)
            for data in stream_resp:
                delta = data.completion
                completion += delta
                if run_manager:
                    run_manager.on_llm_new_token(delta)
        else:
            response = self.client.completions.create(**params)
            completion = response.completion
        return completion

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates a response from the Anthropic model based on the provided messages and parameters."""
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        completion = self._generate_response(params, run_manager)
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """An asynchronous version of `_generate`."""
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        completion = await self._generate_response(params, run_manager)
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Calculates the number of tokens in a given text."""
        if not self.count_tokens:
            raise NameError("Please ensure the anthropic package is loaded")
        return self.count_tokens(text)
