"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_fireworks import ChatFireworks


class TestFireworksStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatFireworks

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)

    @property
    def chat_model_params(self) -> dict:
        return {"api_key": "test_api_key"}
