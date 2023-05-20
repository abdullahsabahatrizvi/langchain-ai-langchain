from __future__ import annotations

import logging
import tempfile
from typing import Dict, Optional

from pydantic import root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureCogsText2SpeechTool(BaseTool):
    """Tool that adds the capability to query the Azure Cognitive Services Text2Speech API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
    """
    azure_cogs_key: str #: :meta private:
    azure_cogs_region: str #: :meta private:
    speech_language: str = "en-US"

    name = "Azure Cognitive Services Text2Speech"
    description = (
        "A wrapper around Azure Cognitive Services Text2Speech. "
        "Useful for when you need to convert text to speech. "
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        azure_cogs_key = get_from_dict_or_env(
            values, "azure_cogs_key", "AZURE_COGS_KEY"
        )
        values["azure_cogs_key"] = azure_cogs_key

        azure_cogs_region = get_from_dict_or_env(
            values, "azure_cogs_region", "AZURE_COGS_REGION"
        )
        values["azure_cogs_region"] = azure_cogs_region

        try:
            import azure.cognitiveservices.speech as speechsdk

        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech is not installed. "
                "Run `pip install azure-cognitiveservices-speech` to install."
            )

        return values

    def _text2speech(self, text: str, speech_language: str) -> str:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            pass
        
        speech_config = speechsdk.SpeechConfig(subscription=self.azure_cogs_key, region=self.azure_cogs_region)
        speech_config.speech_synthesis_language = speech_language

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            stream = speechsdk.AudioDataStream(result)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=f".wav", delete=False) as f:
                stream.save_to_wav_file(f.name)
            
            return f.name
        
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.debug(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                raise RuntimeError(f"Speech synthesis error: {cancellation_details.error_details}")

            return "Speech synthesis canceled."

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            speech_file = self._text2speech(query, self.speech_language)
            return speech_file
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsText2SpeechTool: {e}")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AzureCogsText2SpeechTool does not support async")