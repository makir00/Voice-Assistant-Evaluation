import tempfile
from pathlib import Path

from deepeval.test_case import Audio
from deepeval.voice import CallbackVoiceConnector
from deepeval.voice.connectors import ConnectorTurn

from src.voice.voice_pipeline import VoicePipeline


class VoicePipelineConnector:
    """Adapter between DeepEval voice simulation and VoicePipeline."""

    def __init__(self) -> None:
        self.pipeline = VoicePipeline()

    def process(
        self,
        user_audio: Audio,
    ) -> ConnectorTurn:
        """Process one simulated user audio turn."""

        user_audio_bytes = user_audio.get_bytes()

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:
            temp_audio_path = Path(
                temp_file.name
            )
            temp_file.write(
                user_audio_bytes
            )

        try:
            result = self.pipeline.process_audio(
                str(temp_audio_path)
            )
        finally:
            temp_audio_path.unlink(
                missing_ok=True
            )

        assistant_audio = Audio.from_bytes(
            result["assistant_audio"],
            mimeType="audio/wav",
            sampleRate=24_000,
        )

        return ConnectorTurn(
            audio=assistant_audio,
            transcript=result["assistant_text"],
        )


def create_voice_connector() -> CallbackVoiceConnector:
    """Create the DeepEval connector for the local voice pipeline."""

    adapter = VoicePipelineConnector()

    return CallbackVoiceConnector(
        agent=adapter.process,
    )