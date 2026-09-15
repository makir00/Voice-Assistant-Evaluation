from typing import Any

from src.agent.voice_agent import VoiceAssistant
from src.voice.stt import SpeechToText
from src.voice.tts import TextToSpeech


class VoicePipeline:
    def __init__(self) -> None:
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.agent = VoiceAssistant()

    def process_audio(
        self,
        audio_path: str,
    ) -> dict[str, Any]:

        user_text = self.stt.transcribe(
            audio_path
        )

        agent_result = self.agent.ask(
            user_text
        )

        assistant_audio = self.tts.synthesize(
            agent_result["answer"]
        )

        return {
            "user_text": user_text,
            "assistant_text": agent_result["answer"],
            "tool_calls": agent_result["tool_calls"],
            "tool_results": agent_result["tool_results"],
            "assistant_audio": assistant_audio,
        }