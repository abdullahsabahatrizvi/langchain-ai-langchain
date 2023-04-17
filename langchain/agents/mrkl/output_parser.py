import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"


class MRKLOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        # \s matches against tab/newline/whitespace
        # Action Input section optional to allow for no-input actions.
        regex = r"Action\s*\d*\s*:\s*([ \S]*)(\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*))?$"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(3)
        if action_input:
            action_input = action_input.strip(" ").strip('"').strip("'")
        return AgentAction(action, action_input, text)
