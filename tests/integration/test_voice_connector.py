from pathlib import Path

from deepeval.test_case import Audio
from deepeval.voice import CallbackVoiceConnector
from deepeval.voice.connectors import ConnectorTurn

from src.adapter.voice_connector import (
    VoicePipelineConnector,
    create_voice_connector,
)


FIXTURE_PATH = Path(
    "audio_data/fixtures/main/turn1.wav"
)


def test_voice_pipeline_connector() -> None:
    assert FIXTURE_PATH.exists(), (
        f"Audio fixture not found: {FIXTURE_PATH}"
    )

    user_audio = Audio(
        url=str(FIXTURE_PATH),
    )

    adapter = VoicePipelineConnector()

    result = adapter.process(
        user_audio
    )

    assert isinstance(
        result,
        ConnectorTurn,
    )

    assert isinstance(
        result.audio,
        Audio,
    )

    assert result.audio.mimeType == "audio/wav"

    assert result.transcript is not None
    assert result.transcript.strip()


def test_create_voice_connector() -> None:
    connector = create_voice_connector()

    assert isinstance(
        connector,
        CallbackVoiceConnector,
    )