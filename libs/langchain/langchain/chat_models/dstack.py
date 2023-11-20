from typing import Any, Dict, List, Optional, cast

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.dstack import _BaseDstack
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class ChatDstack(_BaseDstack, BaseChatModel):
    """dstack services API

    To use, you should have `transformers` library.

    Required parameters:
        - `run_name`
        - `api_base_url` or in the environment variable `DSTACK_API_BASE_URL`
        - `api_token` or in the environment variable `DSTACK_API_TOKEN`
        - `project` or in the environment variable `DSTACK_PROJECT`

    Example:
        .. code-block:: python
            from langchain.chat_models import ChatDstack
            llm = ChatDstack(run_name="mistral-7b-gptq")
    """

    @root_validator()
    def require_tokenizer(cls, values: Dict) -> Dict:
        """Check that tokenizer is provided."""
        if values["tokenizer"] is None:
            raise ValueError(
                "Tokenizer is required for chat model."
                " Run `pip install --upgrade transformers`"
            )
        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate the next message in conversation.

        Args:
            messages: The history of the conversation.
            stop: The list of stop sequences.

        Returns:
            The message generated by the model.
        """
        prompt = self.tokenizer.apply_chat_template(
            _get_transformers_conversation(messages), tokenize=False
        )
        resp = self._tgi_generate(prompt, self._tgi_parameters(stop, **kwargs))
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=resp["generated_text"]))
            ]
        )


def _get_transformers_conversation(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert Langchain messages to the format accepted by `transformers`.

    Returns:
        List of messages
    """
    conversation = []
    for message in messages:
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            conversation.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            conversation.append({"role": "assistant", "content": content})
        elif isinstance(message, SystemMessage):
            conversation.append({"role": "system", "content": content})
        else:
            raise ValueError(f"Unsupported message type: {type(message).__name__}")
    return conversation
