import json
from typing import Any, Dict, List

from langchain.pydantic_v1 import BaseModel, Extra


class AnySdkWrapper(BaseModel):
    client: Any
    operations: List[Dict] = []

    class Config:
        extra = Extra.allow
    
    def __init__(self, **data):
        super().__init__(**data)
        self.operations = self._build_operations()

    def _build_operations(self):
        operations = []
        sdk_functions = [
            func
            for func in dir(self.client)
            if callable(getattr(self.client, func)) and not func.startswith("_")
        ]

        for func_name in sdk_functions:
            func = getattr(self.client, func_name)
            operation = {
                "mode": func_name,
                "name": func.__name__.replace("_", " ").title(),
                "description": func.__doc__
            }
            operations.append(operation)
        return operations
    
    def run(self, mode: str, query: str) -> str:
        try:
            params = json.loads(query)
            func = getattr(self.client, mode)
            result = func(**params)
            return json.dumps(result)
        except AttributeError:
            return f"Invalid mode: {mode}"