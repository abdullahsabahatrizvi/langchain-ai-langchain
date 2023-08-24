"""Test the telegram chat loader."""
import pathlib
import tempfile
import zipfile
from typing import Sequence

import pytest

from langchain import schema
from langchain.chat_loaders import telegram, utils


def _assert_messages_are_equal(
    actual_messages: Sequence[schema.BaseMessage],
    expected_messages: Sequence[schema.BaseMessage],
) -> None:
    assert len(actual_messages) == len(expected_messages)
    for actual, expected in zip(actual_messages, expected_messages):
        assert actual.content == expected.content
        assert (
            actual.additional_kwargs["sender"] == expected.additional_kwargs["sender"]
        )


@pytest.mark.parametrize(
    "path",
    [
        "telegram_chat_html",
        "telegram_chat_json",
        "telegram_chat_html.zip",
        "telegram_chat_json.zip",
        "telegram_chat_html/messages.html",
        "telegram_chat_json/result.json",
    ],
)
@pytest.mark.requires("bs4")
def test_telegram_chat_loader(path: str) -> None:
    _data_dir = pathlib.Path(__file__).parent / "data"
    source_path = _data_dir / path
    # Create a zip file from the directory in a temp directory
    with tempfile.TemporaryDirectory() as temp_dir_:
        temp_dir = pathlib.Path(temp_dir_)
        if path.endswith(".zip"):
            # Make a new zip file
            zip_path = temp_dir / "telegram_chat.zip"
            with zipfile.ZipFile(zip_path, "w") as zip_file:
                original_path = _data_dir / path.replace(".zip", "")
                for file_path in original_path.iterdir():
                    zip_file.write(file_path, arcname=file_path.name)
            source_path = zip_path
        loader = telegram.TelegramChatLoader(str(source_path))
        chat_sessions_ = loader.lazy_load()
        chat_sessions_ = utils.merge_chat_runs(chat_sessions_)
        chat_sessions = list(
            utils.map_ai_messages(chat_sessions_, sender="Batman & Robin")
        )
        assert len(chat_sessions) == 1
        session = chat_sessions[0]
        assert len(session["messages"]) > 0
        assert session["messages"][0].content == "i refuse to converse with you"
        expected_content = [
            schema.HumanMessage(
                content="i refuse to converse with you",
                additional_kwargs={
                    "sender": "Jimmeny Marvelton",
                    "events": [{"message_time": "23.08.2023 13:11:23 UTC-08:00"}],
                },
            ),
            schema.AIMessage(
                content="Hi nemesis",
                additional_kwargs={
                    "sender": "Batman & Robin",
                    "events": [{"message_time": "23.08.2023 13:13:20 UTC-08:00"}],
                },
            ),
            schema.HumanMessage(
                content="we meet again\n\nyou will not trick me this time",
                additional_kwargs={
                    "sender": "Jimmeny Marvelton",
                    "events": [{"message_time": "23.08.2023 13:15:35 UTC-08:00"}],
                },
            ),
        ]
        _assert_messages_are_equal(session["messages"], expected_content)
