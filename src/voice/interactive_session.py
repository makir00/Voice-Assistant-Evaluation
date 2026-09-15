from pathlib import Path
from typing import Any

from src.voice.microphone import Microphone
from src.voice.speaker import Speaker
from src.voice.voice_pipeline import VoicePipeline


INTERACTIVE_AUDIO_DIR = Path(
    "audio_data/interactive"
)


def run_interactive_session() -> list[dict[str, Any]]:
    pipeline = VoicePipeline()
    microphone = Microphone()
    speaker = Speaker()

    INTERACTIVE_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conversation_trace = []
    turn_number = 1

    print("Voice assistant started.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            input("Press ENTER to record...")

            audio_path = (
                INTERACTIVE_AUDIO_DIR
                / f"interactive_input_turn_{turn_number}.wav"
            )

            microphone.record(
                duration=3,
                output_path=str(audio_path),
            )

            result = pipeline.process_audio(
                str(audio_path)
            )

            print(
                f"\nUSER: {result['user_text']}"
            )
            print(
                f"ASSISTANT: {result['assistant_text']}"
            )
            print(
                f"TOOLS: {result['tool_calls']}"
            )

            conversation_trace.append(
                result
            )

            speaker.play(
                result["assistant_audio"]
            )

            turn_number += 1

    except KeyboardInterrupt:
        print(
            "\nVoice assistant stopped."
        )

    return conversation_trace