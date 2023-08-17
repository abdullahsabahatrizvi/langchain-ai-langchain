from __future__ import annotations
import logging
from typing import Dict, Optional,Any,Literal
from pydantic import root_validator,Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env
import requests
from langchain.tools.edenai.edenai_base_tool import EdenaiTool
logger = logging.getLogger(__name__)
  

class EdenAiTextToSpeech(EdenaiTool):
    """Tool that queries the Eden AI Text to speech API.
    for api reference check edenai documentation: https://docs.edenai.co/reference/audio_text_to_speech_create.
    
    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    edenai_api_key: Optional[str] = None

    name="edenai_text_to_speech"
    description = (
        "A wrapper around edenai Services text to speech."
        "Useful for when you need to convert text to speech."
        """the output is a string representing the URL of the audio file,
        or the path to the downloaded wav file """
    )
    
    
    params: Optional[Dict[str,Any]] = Field(default_factory=dict)
    
    language: Optional[str] = "en"
    """
    language of the text passed to the model.
    """    

    voice : Literal["MALE","FEMALE"] 
    """voice option : 'MALE' or 'FEMALE' """

    feature: str="audio"
    subfeature : str ="text_to_speech"


    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values


    def _download_wav(self,url : str, save_path: str) -> None:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"Error while downloading wav file" )
        
    def _format_text_to_speech_result(self, text_to_speech_result: list) -> str:
        result=text_to_speech_result[0]
        if self.params["return_type"] == "url":
            return result["audio_resource_url"]
        elif self.params["return_type"] == "wav":
            self._download_wav(result["audio_resource_url"], "audio.wav")
            return "audio.wav"
        else :
            raise ValueError(f"return_type should be url or wav, not {self.params['return_type']}")

    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:

            if "return_type" not in self.params.keys() :
                self.params["return_type"] = "url"
            query_params = {"text": query, "language": self.language ,"option": self.voice,**self.params}
            
            text_analysis_result = self._call_eden_ai(query_params)
            
            text_analysis_result=text_analysis_result.json()
            
            result = self._format_text_to_speech_result(text_analysis_result)
            print(result)
            return result
        
        except Exception as e:
            raise RuntimeError(f"Error while running Edenai API: {e}")
