import uuid
from operator import itemgetter

import pytest
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.hunyuan import ChatHunyuan


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan() -> None:
    chat = ChatHunyuan()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.id) > 0, "request_id 不为空"
    assert uuid.UUID(response.id), "无效的UUID"


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_temperature() -> None:
    chat = ChatHunyuan(temperature=0.6)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.id) > 0, "request_id 不为空"
    assert uuid.UUID(response.id), "无效的UUID"


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_model_name() -> None:
    chat = ChatHunyuan(model="hunyuan-standard")
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.id) > 0, "request_id 不为空"
    assert uuid.UUID(response.id), "无效的UUID"


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_stream() -> None:
    chat = ChatHunyuan(streaming=True)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.id) > 0, "request_id 不为空"
    assert uuid.UUID(response.id), "无效的UUID"


@pytest.mark.requires("tencentcloud-sdk-python")
def test_chat_hunyuan_with_prompt_template() -> None:
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant! Your name is {name}."
    )
    user_prompt = HumanMessagePromptTemplate.from_template("Question: {query}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    chat = (
        {"query": itemgetter("query"), "name": itemgetter("name")}
        | chat_prompt
        | ChatHunyuan()
    )
    response = chat.invoke({"query": "Hello", "name": "Tom"})
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.id) > 0, "request_id 不为空"
    assert uuid.UUID(response.id), "无效的UUID"


def test_extra_kwargs() -> None:
    chat = ChatHunyuan(temperature=0.88, top_p=0.7)
    assert chat.temperature == 0.88
    assert chat.top_p == 0.7
