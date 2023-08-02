"""Memory used to save agent output AND intermediate steps."""
from typing import Any, Dict, List

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.agents.openai_functions_agent.base import _format_intermediate_steps


class AgentTokenBufferMemory(BaseChatMemory):
    """Memory used to save agent output AND intermediate steps."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 12000
    """The max number of tokens to keep in the buffer. 
    Once the buffer exceeds this many tokens, the oldest messages will be pruned."""
    return_messages: bool = True
    output_key = "output"
    intermediate_steps_key = "intermediate_steps"

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        steps = _format_intermediate_steps(outputs["intermediate_steps"])
        for msg in steps:
            self.chat_memory.add_message(msg)
        self.chat_memory.add_ai_message(output_str)
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)