import os
from typing import Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Field, validator
from mixedbread_ai.client import AsyncMixedbreadAI, MixedbreadAI  # type: ignore
from mixedbread_ai.core import RequestOptions  # type: ignore


class MixedBreadAIClient(BaseModel):
    _client: MixedbreadAI = Field(default=None, exclude=True)
    _aclient: AsyncMixedbreadAI = Field(default=None, exclude=True)
    _request_options: Optional[RequestOptions] = Field(default=None, exclude=True)

    api_key: str = Field(
        alias="mxbai_api_key",
        default_factory=lambda: os.environ.get("MXBAI_API_KEY", None),
        description="mixedbread ai API key. Must be specified directly or "
        "via environment variable 'MXBAI_API_KEY'",
        min_length=1,
    )
    base_url: Optional[str] = Field(
        alias="mxbai_api_base",
        default=None,
        description="Base URL for the mixedbread ai API. "
        "Leave blank if not using a proxy or service emulator.",
        min_length=1,
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for the mixedbread ai API",
        ge=0,
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="Max retries for the mixedbread ai API",
        ge=0,
    )

    def __init__(self, **data: Dict) -> None:
        super().__init__(**data)

        object.__setattr__(
            self,
            "_client",
            MixedbreadAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            ),
        )
        object.__setattr__(
            self,
            "_aclient",
            AsyncMixedbreadAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            ),
        )
        object.__setattr__(
            self,
            "_request_options",
            (
                RequestOptions(max_retries=self.max_retries)
                if self.max_retries is not None
                else None
            ),
        )

    @validator("api_key")
    def validate_api_key(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "The mixedbread ai API key must be specified."
                + "You either pass it in the constructor using 'mxbai_api_key'"
                + "or via the 'MXBAI_API_KEY' environment variable."
            )
        return value
