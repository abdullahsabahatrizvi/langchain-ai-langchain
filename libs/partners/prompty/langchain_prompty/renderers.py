from langchain_core.utils import mustache
from pydantic import BaseModel

from .core import Invoker, InvokerFactory, Prompty, SimpleModel


class MustacheRenderer(Invoker):
    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty

    def invoke(self, data: BaseModel) -> BaseModel:
        assert isinstance(data, SimpleModel)
        generated = mustache.render(self.prompty.content, data.item)
        return SimpleModel[str](item=generated)
