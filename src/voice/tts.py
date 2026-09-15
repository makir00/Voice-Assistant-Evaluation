import io
import os
import wave

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class TextToSpeech:
    """Convert assistant text responses into WAV audio."""

    SAMPLE_RATE = 24_000
    CHANNELS = 1
    SAMPLE_WIDTH = 2

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
            "OPENAI_TTS_MODEL",
            "gpt-4o-mini-tts",
        )

        self.voice = os.getenv(
            "OPENAI_TTS_VOICE",
            "nova",
        )

    def synthesize(
        self,
        text: str,
    ) -> bytes:
        """Convert text into mono 24 kHz PCM WAV audio."""

        if not text.strip():
            raise ValueError(
                "TTS input text cannot be empty."
            )

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
            speed=1.0,
            instructions=(
                "Speak in a warm, bright, clearly feminine voice. "
                "Use a moderately higher vocal register while remaining "
                "natural. "
                "Speak clearly at a steady conversational pace. "
                "Keep pauses very short and consistent. "
                "Do not use dramatic pauses, hesitation sounds, "
                "drawn-out words, or a deep vocal register. "
                "Start speaking promptly and finish cleanly."
            ),
        )

        pcm_audio = response.read()

        if not pcm_audio:
            raise RuntimeError(
                "The TTS API returned empty audio data."
            )

        return self._pcm_to_wav(
            pcm_audio
        )

    @classmethod
    def _pcm_to_wav(
        cls,
        pcm_audio: bytes,
    ) -> bytes:
        """Wrap raw PCM bytes in a WAV container."""

        wav_buffer = io.BytesIO()

        with wave.open(
            wav_buffer,
            "wb",
        ) as wav_file:
            wav_file.setnchannels(
                cls.CHANNELS
            )
            wav_file.setsampwidth(
                cls.SAMPLE_WIDTH
            )
            wav_file.setframerate(
                cls.SAMPLE_RATE
            )
            wav_file.writeframes(
                pcm_audio
            )

        return wav_buffer.getvalue()