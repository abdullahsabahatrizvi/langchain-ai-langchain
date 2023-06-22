import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    LLMResult,
    SystemMessage,
)


class PromptLayerCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        pl_id_callback: Optional[Callable[..., Any]] = None,
        pl_tags: Optional[List[str]] = [],
    ) -> None:
        self.pl_id_callback = pl_id_callback
        self.pl_tags = pl_tags

        self.runs: Dict[UUID, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        self.runs[run_id] = {
            "messages": [self._create_message_dicts(m) for m in messages],
            "invocation_params": kwargs.get("invocation_params", {}),
            "name": kwargs.get("invocation_params", {}).get("_type", "No Type"),
            "request_start_time": datetime.datetime.now().timestamp(),
            "tags": tags,
        }

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        self.runs[run_id] = {
            "prompts": prompts,
            "invocation_params": kwargs.get("invocation_params", {}),
            "name": kwargs.get("invocation_params", {}).get("_type", "No Type"),
            "request_start_time": datetime.datetime.now().timestamp(),
            "tags": tags,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        run_info = self.runs.get(run_id, {})
        run_info["request_end_time"] = datetime.datetime.now().timestamp()
        if not run_info:
            return
        for i in range(len(response.generations)):
            generation = response.generations[i][0]

            resp = {
                "text": generation.text,
                "llm_output": response.llm_output,
            }
            model_params = run_info.get("invocation_params", {})
            if run_info.get("name") == "openai-chat":
                function_name = f"langchain.chat.{run_info.get('name')}"
                model_input = run_info.get("messages", [])[i]
                model_response = self._convert_message_to_dict(generation.message)
            else:
                function_name = f"langchain.{run_info.get('name')}"
                model_input = [run_info.get("prompts", [])[i]]
                model_response = resp

            from promptlayer.utils import get_api_key, promptlayer_api_request

            pl_request_id = promptlayer_api_request(
                function_name,
                "langchain",
                model_input,
                model_params,
                self.pl_tags,
                model_response,
                run_info.get("request_start_time"),
                run_info.get("request_end_time"),
                get_api_key(),
                return_pl_id=bool(self.pl_id_callback != None),
                metadata={
                    "_langchain_run_id": str(run_id),
                    "_langchain_parent_run_id": str(parent_run_id),
                    "_langchain_tags": str(run_info.get("tags", [])),
                },
            )

            if self.pl_id_callback:
                self.pl_id_callback(pl_request_id)

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params
