from typing import Any, Dict, List

from pydantic import BaseModel, root_validator

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import (
    AIMessage,
    BaseLanguageModel,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)


class SummarizerMixin(BaseModel):
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT

    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)


class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin, BaseModel):
    """Conversation summarizer to memory."""

    buffer: str = ""
    summary_message_role: str = "system"
    memory_key: str = "history"  #: :meta private:

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        if self.return_messages:
            if self.summary_message_role == "user":
                buffer: Any = [HumanMessage(content=self.buffer)]
            elif self.summary_message_role == "assistant":
                buffer: Any = [AIMessage(content=self.buffer)]
            elif self.summary_message_role == "system":
                buffer: Any = [SystemMessage(content=self.buffer)]
            else:
                raise ValueError(
                    f"Invalid summary_message_role value: {self.summary_message_role}."
                    " summary_message_role must be 'user', 'assistant', or 'system'."
                )
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:], self.buffer
        )

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""
