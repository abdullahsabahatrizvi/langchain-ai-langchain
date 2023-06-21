"""Util that Searches email messages in Office 365.

Free, but setup is required. See link below.
https://learn.microsoft.com/en-us/graph/auth/
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Extra, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.office365.base import O365BaseTool
from langchain.tools.office365.utils import clean_body


class SearchEmailsInput(BaseModel):
    """Input for SearchEmails Tool."""

    """From https://learn.microsoft.com/en-us/graph/search-query-parameter"""

    folder: str = Field(
        default=None,
        description=" If the user wants to search in only one folder, the name of the folder. Default folders"
        ' are "inbox", "drafts", "sent items", "deleted ttems", but users can search custom folders as well.',
    )
    query: str = Field(
        description="The Microsoift Graph v1.0 $search query. Example filters include from:sender,"
        " from:sender, to:recipient, subject:subject, recipients:list_of_recipients,"
        " body:excitement, importance:high, received>2022-12-01, received<2021-12-01,"
        " sent>2022-12-01, sent<2021-12-01, hasAttachments:true"
        " attachment:api-catalog.md, cc:samanthab@contoso.com, bcc:samanthab@contoso.com, body:excitement"
        " date range example: received:2023-06-08..2023-06-09"
        " matching example: from:amy OR from:david.",
    )
    max_results: int = Field(
        default=10,
        description="The maximum number of results to return.",
    )
    truncate: bool = Field(
        default=True,
        description="Whether the email body is trucated to meet token number limits. Set to False for"
        " searches that will retrieve very few results, otherwise, set to True",
    )


class O365SearchEmails(O365BaseTool):
    """Class for searching email messages in Office 365

    Free, but setup is required
    """

    name: str = "messages_search"
    args_schema: Type[BaseModel] = SearchEmailsInput
    description: str = (
        "Use this tool to search for email messages."
        " The input must be a valid Microsoft Graph v1.0 $search query."
        " The output is a JSON list of the requested resource."
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _run(
        self,
        query: str,
        folder: str = "",
        max_results: int = 10,
        truncate: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        # Get mailbox object
        mailbox = self.account.mailbox()

        # Pull the folder if the user wants to search in a folder
        if folder is not "":
            mailbox = mailbox.get_folder(folder_name=folder)

        # Retrieve messages based on query
        query = mailbox.q().search(query)
        messages = mailbox.get_messages(limit=max_results, query=query)

        # Generate output dict
        output_messages = []
        for message in messages:
            output_message = {}
            output_message["from"] = message.sender

            if truncate:
                output_message["body"] = message.body_preview
            else:
                output_message["body"] = clean_body(message.body)

            output_message["subject"] = message.subject

            output_message["date"] = message.modified.strftime("%Y-%m-%dT%H:%M:%S%z")

            output_message["to"] = []
            for recipient in message.to._recipients:
                output_message["to"].append(str(recipient))

            output_message["cc"] = []
            for recipient in message.cc._recipients:
                output_message["cc"].append(str(recipient))

            output_message["bcc"] = []
            for recipient in message.bcc._recipients:
                output_message["bcc"].append(str(recipient))

            output_messages.append(output_message)

        return output_messages

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool."""
        raise NotImplementedError
