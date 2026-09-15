from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import PromptAlignmentMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from src.voice.voice_pipeline import VoicePipeline


FIXTURE_DIR = Path(
    "audio data/fixtures/main"
)

PROMPT_INSTRUCTIONS = [
    (
        "Act as a helpful and friendly travel voice assistant."
    ),
    (
        "Help with travel-related requests such as weather, "
        "attractions, restaurants, and practical travel recommendations."
    ),
    (
        "Politely decline requests that are outside the travel "
        "assistance domain."
    ),
    (
        "Keep responses concise and natural for spoken conversation."
    ),
]


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def create_prompt_alignment_test_cases() -> list[LLMTestCase]:
    pipeline = VoicePipeline()

    audio_turns = sorted(
        FIXTURE_DIR.glob("turn*.wav")
    )

    if not audio_turns:
        raise FileNotFoundError(
            f"No audio fixtures found in: {FIXTURE_DIR}"
        )

    test_cases = []

    for turn_number, audio_path in enumerate(
        audio_turns,
        start=1,
    ):
        result = pipeline.process_audio(
            str(audio_path)
        )

        print(
            f"\n--- TURN {turn_number} ---"
        )
        print(
            f"AUDIO FIXTURE: {audio_path.name}"
        )
        print(
            f"USER INPUT: "
            f"{result['user_text']}"
        )
        print(
            f"ASSISTANT OUTPUT: "
            f"{result['assistant_text']}"
        )

        test_cases.append(
            LLMTestCase(
                input=result["user_text"],
                actual_output=result["assistant_text"],
            )
        )

    return test_cases


def test_prompt_alignment() -> None:
    dataset = EvaluationDataset()

    test_cases = (
        create_prompt_alignment_test_cases()
    )

    for test_case in test_cases:
        dataset.add_test_case(
            test_case
        )

    metric = PromptAlignmentMetric(
        prompt_instructions=PROMPT_INSTRUCTIONS,
        threshold=0.75,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[metric],
    )