from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import yaml
from pydantic import Field, root_validator

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage
from langchain.schema.output_parser import BaseOutputParser


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""


class BasePromptTemplate(Serializable, ABC):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Union[str, Callable[[], str]]] = Field(
        default_factory=dict
    )

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""

    @root_validator()
    def validate_variable_names(cls, values: Dict) -> Dict:
        """Validate variable names do not include restricted names."""
        if "stop" in values["input_variables"]:
            raise ValueError(
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
        if "stop" in values["partial_variables"]:
            raise ValueError(
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )

        overall = set(values["input_variables"]).intersection(
            values["partial_variables"]
        )
        if overall:
            raise ValueError(
                f"Found overlapping input and partial variables: {overall}"
            )
        return values

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if isinstance(v, str) else v()
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of prompt."""
        prompt_dict = super().dict(**kwargs)
        prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Example:
        .. code-block:: python

            prompt.save(file_path="path/prompt.yaml")
        """
        if self.partial_variables:
            raise ValueError("Cannot save prompt with partial variables.")
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")
