# Records microphone input and saves each conversation turn as a WAV fixture for evaluation tests.

import argparse
from pathlib import Path

from src.voice.microphone import Microphone


FIXTURES_DIR = Path("audio data/fixtures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio data turns for a test fixture."
    )

    parser.add_argument(
        "fixture_name",
        help=(
            "Fixture directory name, for example: "
            "main, goal_accuracy, knowledge_retention"
        ),
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=4,
        help="Number of conversation turns to record.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=8,
        help="Recording duration per turn in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fixture_dir = FIXTURES_DIR / args.fixture_name
    fixture_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    microphone = Microphone()

    for turn_number in range(1, args.turns + 1):
        output_path = (
            fixture_dir / f"turn{turn_number}.wav"
        )

        input(
            f"Press ENTER to record TURN {turn_number}..."
        )

        microphone.record(
            duration=args.duration,
            output_path=str(output_path),
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()