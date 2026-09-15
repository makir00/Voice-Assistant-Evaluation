from pathlib import Path

from src.voice.voice_pipeline import VoicePipeline


FIXTURE_DIR = Path(
    "audio data/fixtures/main"
)


def test_multi_turn_voice_pipeline() -> None:
    pipeline = VoicePipeline()

    audio_turns = sorted(
        FIXTURE_DIR.glob("turn*.wav")
    )

    assert audio_turns, (
        f"No audio fixtures found in: {FIXTURE_DIR}"
    )

    for index, audio_path in enumerate(
        audio_turns,
        start=1,
    ):
        result = pipeline.process_audio(
            str(audio_path)
        )

        assert result["user_text"].strip()
        assert result["assistant_text"].strip()

        assert isinstance(
            result["tool_calls"],
            list,
        )

        assert isinstance(
            result["tool_results"],
            list,
        )

        assistant_audio = result[
            "assistant_audio"
        ]

        assert isinstance(
            assistant_audio,
            bytes,
        )

        assert assistant_audio

        assert assistant_audio.startswith(
            b"RIFF"
        )

        assert assistant_audio[8:12] == b"WAVE"

        print(
            f"\n--- TURN {index} ---"
        )
        print(
            f"USER: {result['user_text']}"
        )
        print(
            f"ASSISTANT: "
            f"{result['assistant_text']}"
        )

        tool_names = [
            tool_call["name"]
            for tool_call in result["tool_calls"]
        ]

        print(
            f"TOOLS: "
            f"{tool_names or ['None']}"
        )
        print(
            f"AUDIO BYTES: "
            f"{len(assistant_audio)}"
        )