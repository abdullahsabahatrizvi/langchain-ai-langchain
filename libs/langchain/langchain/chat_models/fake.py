"""Fake ChatModel for testing purposes."""
import asyncio
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import AIMessageChunk, BaseMessage, AIMessage
from langchain.schema.output import ChatGenerationChunk, ChatGeneration, ChatResult


class FakeListChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    responses: List
    sleep: Optional[float] = None
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                time.sleep(self.sleep)
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"responses": self.responses}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = next(self.responses)  # type: ignore

        if isinstance(response, dict):
            message = AIMessage(
                content="",
                additional_kwargs={"function_call": response},
            )
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        return super()._generate(messages, stop, run_manager, **kwargs)
