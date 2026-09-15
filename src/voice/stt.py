import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class SpeechToText:
    """Convert speech audio into text."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = OpenAI(
            api_key=api_key,
        )

        self.model = os.getenv(
            "OPENAI_STT_MODEL",
            "gpt-4o-mini-transcribe",
        )

    def transcribe(
        self,
        audio_path: str,
    ) -> str:
        """Transcribe a WAV audio file into text."""

        with open(
            audio_path,
            "rb",
        ) as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )

        return response.text