from typing import Any, AsyncIterator, Iterator, List

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field

STREAMED_MESSAGES: list = [
    AIMessageChunk(content=""),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
                    "function": {"arguments": "", "name": "NameCollector"},
                    "type": "function",
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": '{"na', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'mes":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' ["suz', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'y", ', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": '"jerm', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'aine",', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "al', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'ex"],', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "pers', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'on":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' {"ag', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'e": 39', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ', "h', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": "air_c", "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'olor":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "br', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'own",', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "job"', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ': "c', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": "oncie", "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'rge"}}', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(content=""),
]


STREAMED_MESSAGES_WITH_TOOL_CALLS = []
for message in STREAMED_MESSAGES:
    if message.additional_kwargs:
        STREAMED_MESSAGES_WITH_TOOL_CALLS.append(
            AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                tool_call_chunks=[
                    ToolCallChunk(
                        name=chunk["function"].get("name"),
                        args=chunk["function"].get("arguments"),
                        id=chunk.get("id"),
                        index=chunk["index"],
                    )
                    for chunk in message.additional_kwargs["tool_calls"]
                ],
            )
        )
    else:
        STREAMED_MESSAGES_WITH_TOOL_CALLS.append(message)


EXPECTED_STREAMED_JSON = [
    {},
    {"names": ["suz"]},
    {"names": ["suzy"]},
    {"names": ["suzy", "jerm"]},
    {"names": ["suzy", "jermaine"]},
    {"names": ["suzy", "jermaine", "al"]},
    {"names": ["suzy", "jermaine", "alex"]},
    {"names": ["suzy", "jermaine", "alex"], "person": {}},
    {"names": ["suzy", "jermaine", "alex"], "person": {"age": 39}},
    {"names": ["suzy", "jermaine", "alex"], "person": {"age": 39, "hair_color": "br"}},
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "c"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "concie"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "concierge"},
    },
]


def _get_iter(use_tool_calls: bool = False) -> Any:
    if use_tool_calls:
        list_to_iter = STREAMED_MESSAGES_WITH_TOOL_CALLS
    else:
        list_to_iter = STREAMED_MESSAGES

    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in list_to_iter:
            yield msg

    return input_iter


def _get_aiter(use_tool_calls: bool = False) -> Any:
    if use_tool_calls:
        list_to_iter = STREAMED_MESSAGES_WITH_TOOL_CALLS
    else:
        list_to_iter = STREAMED_MESSAGES

    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for msg in list_to_iter:
            yield msg

    return input_iter


def test_partial_json_output_parser() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls)
        chain = input_iter | JsonOutputToolsParser()

        actual = list(chain.stream(None))
        expected: list = [[]] + [
            [{"type": "NameCollector", "args": chunk}]
            for chunk in EXPECTED_STREAMED_JSON
        ]
        assert actual == expected


async def test_partial_json_output_parser_async() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_aiter(use_tool_calls)
        chain = input_iter | JsonOutputToolsParser()

        actual = [p async for p in chain.astream(None)]
        expected: list = [[]] + [
            [{"type": "NameCollector", "args": chunk}]
            for chunk in EXPECTED_STREAMED_JSON
        ]
        assert actual == expected


def test_partial_json_output_parser_return_id() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls)
        chain = input_iter | JsonOutputToolsParser(return_id=True)

        actual = list(chain.stream(None))
        expected: list = [[]] + [
            [
                {
                    "type": "NameCollector",
                    "args": chunk,
                    "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
                }
            ]
            for chunk in EXPECTED_STREAMED_JSON
        ]
        assert actual == expected


def test_partial_json_output_key_parser() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls)
        chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

        actual = list(chain.stream(None))
        expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
        assert actual == expected


async def test_partial_json_output_parser_key_async() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_aiter(use_tool_calls)

        chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

        actual = [p async for p in chain.astream(None)]
        expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
        assert actual == expected


def test_partial_json_output_key_parser_first_only() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls)

        chain = input_iter | JsonOutputKeyToolsParser(
            key_name="NameCollector", first_tool_only=True
        )

        assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON


async def test_partial_json_output_parser_key_async_first_only() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_aiter(use_tool_calls)

        chain = input_iter | JsonOutputKeyToolsParser(
            key_name="NameCollector", first_tool_only=True
        )

        assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON


class Person(BaseModel):
    age: int
    hair_color: str
    job: str


class NameCollector(BaseModel):
    """record names of all people mentioned"""

    names: List[str] = Field(..., description="all names mentioned")
    person: Person = Field(..., description="info about the main subject")


# Expected to change when we support more granular pydantic streaming.
EXPECTED_STREAMED_PYDANTIC = [
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="c"),
    ),
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="concie"),
    ),
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="concierge"),
    ),
]


def test_partial_pydantic_output_parser() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls)

        chain = input_iter | PydanticToolsParser(
            tools=[NameCollector], first_tool_only=True
        )

        actual = list(chain.stream(None))
        assert actual == EXPECTED_STREAMED_PYDANTIC


async def test_partial_pydantic_output_parser_async() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_aiter(use_tool_calls)

        chain = input_iter | PydanticToolsParser(
            tools=[NameCollector], first_tool_only=True
        )

        actual = [p async for p in chain.astream(None)]
        assert actual == EXPECTED_STREAMED_PYDANTIC
