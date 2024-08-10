"""Test OpenAI Chat API wrapper."""

import json
from typing import Any, Dict, List, Literal, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    _format_message_content,
)


def test_openai_model_param() -> None:
    llm = ChatOpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = ChatOpenAI(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {"role": "function", "name": name, "content": content}
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_human_with_name() -> None:
    message = {"role": "user", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_ai_with_name() -> None:
    message = {"role": "assistant", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
        "type": "function",
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message

    # Test malformed tool call
    raw_tool_calls: list = [
        {
            "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
            "function": {"arguments": "oops", "name": "GenerateUsername"},
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"name": "Sally", "hair_color": "green"}',
                "name": "GenerateUsername",
            },
            "type": "function",
        },
    ]
    raw_tool_calls = list(sorted(raw_tool_calls, key=lambda x: x["id"]))
    message = {"role": "assistant", "content": None, "tool_calls": raw_tool_calls}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)",  # noqa: E501
                type="invalid_tool_call",
            )
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_abc123",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    reverted_message_dict = _convert_message_to_dict(expected_output)
    reverted_message_dict["tool_calls"] = list(
        sorted(reverted_message_dict["tool_calls"], key=lambda x: x["id"])
    )
    assert reverted_message_dict == message


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Bar Baz", "name": "Erick"},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def mock_client(mock_completion: dict) -> MagicMock:
    rtn = MagicMock()

    mock_create = MagicMock()

    mock_resp = MagicMock()
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.parse.return_value = mock_completion
    mock_create.return_value = mock_resp

    rtn.with_raw_response.create = mock_create
    rtn.create.return_value = mock_completion
    return rtn


@pytest.fixture
def mock_async_client(mock_completion: dict) -> AsyncMock:
    rtn = AsyncMock()

    mock_create = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.parse.return_value = mock_completion
    mock_create.return_value = mock_resp

    rtn.with_raw_response.create = mock_create
    rtn.create.return_value = mock_completion
    return rtn


def test_openai_invoke(mock_client: MagicMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "client", mock_client):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"

        # headers are not in response_metadata if include_response_headers not set
        assert "headers" not in res.response_metadata
    assert mock_client.create.called


async def test_openai_ainvoke(mock_async_client: AsyncMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "async_client", mock_async_client):
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"

        # headers are not in response_metadata if include_response_headers not set
        assert "headers" not in res.response_metadata
    assert mock_async_client.create.called


@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-3.5-0125",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview",
    ],
)
def test__get_encoding_model(model: str) -> None:
    ChatOpenAI(model=model)._get_encoding_model()
    return


def test_openai_invoke_name(mock_client: MagicMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "client", mock_client):
        messages = [HumanMessage(content="Foo", name="Katie")]
        res = llm.invoke(messages)
        call_args, call_kwargs = mock_client.create.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Katie"

        # check return type has name
        assert res.content == "Bar Baz"
        assert res.name == "Erick"


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> List[int]:
        return [1, 2, 3]

    llm = ChatOpenAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]


def test_format_message_content() -> None:
    content: Any = "hello"
    assert content == _format_message_content(content)

    content = None
    assert content == _format_message_content(content)

    content = []
    assert content == _format_message_content(content)

    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "url.com"}},
    ]
    assert content == _format_message_content(content)

    content = [
        {"type": "text", "text": "hello"},
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA", "unit": "celsius"},
        },
    ]
    assert [{"type": "text", "text": "hello"}] == _format_message_content(content)


class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


class MakeASandwich(BaseModel):
    "Make a sandwich given a list of ingredients."

    bread_type: str
    cheese_type: str
    condiments: List[str]
    vegetables: List[str]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "any",
        "none",
        "auto",
        "required",
        "GenerateUsername",
        {"type": "function", "function": {"name": "MakeASandwich"}},
        False,
        None,
    ],
)
@pytest.mark.parametrize("strict", [True, False, None])
def test_bind_tools_tool_choice(tool_choice: Any, strict: Optional[bool]) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm.bind_tools(
        tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice, strict=strict
    )


@pytest.mark.parametrize("schema", [GenerateUsername, GenerateUsername.schema()])
@pytest.mark.parametrize("method", ["json_schema", "function_calling", "json_mode"])
@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("strict", [True, False, None])
def test_with_structured_output(
    schema: Union[Type, Dict[str, Any], None],
    method: Literal["function_calling", "json_mode", "json_schema"],
    include_raw: bool,
    strict: Optional[bool],
) -> None:
    """Test passing in manually construct tool call message."""
    if method == "json_mode":
        strict = None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm.with_structured_output(
        schema, method=method, strict=strict, include_raw=include_raw
    )

class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool for testing"
    def _run(self):
        return "Dummy result"

@pytest.mark.parametrize("schema", [GenerateUsername, GenerateUsername.schema()])
@pytest.mark.parametrize("method", ["json_schema", "function_calling", "json_mode"])
@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("strict", [True, False, None])
@pytest.mark.parametrize("tools", [None, [DummyTool()]])
def test_with_structured_output_and_tools(
    schema: Union[Type, Dict[str, Any]],
    method: Literal["function_calling", "json_mode", "json_schema"],
    include_raw: bool,
    strict: Optional[bool],
    tools: Optional[List[BaseTool]],
) -> None:
    """Test with_structured_output method with tools."""
    if method == "json_mode":
        strict = None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if method == "json_schema" and tools and strict is False:
        with pytest.raises(ValueError, match="strict must be True when binding tools with method='json_schema'."):
            llm.with_structured_output(
                schema, method=method, strict=strict, include_raw=include_raw, tools=tools
            )
    else:
        structured_llm = llm.with_structured_output(
            schema, method=method, strict=strict, include_raw=include_raw, tools=tools
        )
        assert structured_llm is not None

def test_with_structured_output_json_schema_strict():
    """Test with_structured_output method with json_schema and strict mode."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="json_schema", strict=True
    )
    assert structured_llm is not None

def test_with_structured_output_function_calling_with_tools():
    """Test with_structured_output method with function_calling and tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="function_calling", tools=[DummyTool()]
    )
    assert structured_llm is not None

def test_with_structured_output_json_mode_with_tools():
    """Test with_structured_output method with json_mode and tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="json_mode", tools=[DummyTool()]
    )
    assert structured_llm is not None


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "A dummy tool for testing"

    def _run(self) -> str:
        return "Dummy result"


@pytest.mark.parametrize("schema", [GenerateUsername, GenerateUsername.schema()])
@pytest.mark.parametrize("method", ["json_schema", "function_calling", "json_mode"])
@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("strict", [True, False, None])
@pytest.mark.parametrize("tools", [None, [DummyTool()]])
def test_with_structured_output_and_tools(
    schema: Union[Type, Dict[str, Any]],
    method: Literal["function_calling", "json_mode", "json_schema"],
    include_raw: bool,
    strict: Optional[bool],
    tools: Optional[List[BaseTool]],
) -> None:
    """Test with_structured_output method with tools."""
    if method == "json_mode":
        strict = None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if method == "json_schema" and tools and strict is False:
        with pytest.raises(
            ValueError,
            match="strict must be True when binding tools with method='json_schema'.",
        ):
            llm.with_structured_output(
                schema,
                method=method,
                strict=strict,
                include_raw=include_raw,
                tools=tools,
            )
    else:
        structured_llm = llm.with_structured_output(
            schema, method=method, strict=strict, include_raw=include_raw, tools=tools
        )
        assert structured_llm is not None


def test_with_structured_output_json_schema_strict() -> None:
    """Test with_structured_output method with json_schema and strict mode."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="json_schema", strict=True
    )
    assert structured_llm is not None


def test_with_structured_output_function_calling_with_tools() -> None:
    """Test with_structured_output method with function_calling and tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="function_calling", tools=[DummyTool()]
    )
    assert structured_llm is not None


def test_with_structured_output_json_mode_with_tools() -> None:
    """Test with_structured_output method with json_mode and tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(
        GenerateUsername, method="json_mode", tools=[DummyTool()]
    )
    assert structured_llm is not None


def test_get_num_tokens_from_messages() -> None:
    llm = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage("you're a good assistant"),
        HumanMessage("how are you"),
        HumanMessage(
            [
                {"type": "text", "text": "what's in this image"},
                {"type": "image_url", "image_url": {"url": "https://foobar.com"}},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://foobar.com", "detail": "low"},
                },
            ]
        ),
        AIMessage("a nice bird"),
        AIMessage(
            "",
            tool_calls=[
                ToolCall(id="foo", name="bar", args={"arg1": "arg1"}, type="tool_call")
            ],
        ),
        AIMessage(
            "",
            additional_kwargs={
                "function_call": json.dumps({"arguments": "old", "name": "fun"})
            },
        ),
        AIMessage(
            "text",
            tool_calls=[
                ToolCall(id="foo", name="bar", args={"arg1": "arg1"}, type="tool_call")
            ],
        ),
        ToolMessage("foobar", tool_call_id="foo"),
    ]
    expected = 170
    actual = llm.get_num_tokens_from_messages(messages)
    assert expected == actual
