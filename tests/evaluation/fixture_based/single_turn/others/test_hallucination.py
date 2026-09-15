from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import HallucinationMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from src.agent.voice_agent import VoiceAssistant


judge_model = OpenAIModel(
    model="gpt-5.4",
    temperature=0,
)


def test_hallucination() -> None:
    dataset = EvaluationDataset(
        goldens=[
            Golden(
                input="What's the weather in Munich?",
                context=[
                    "The weather in Munich is sunny and 22°C."
                ],
            ),
            Golden(
                input="What's the weather in Berlin?",
                context=[
                    "The weather in Berlin is cloudy and 19°C."
                ],
            ),
            Golden(
                input="What's the weather in Vienna?",
                context=[
                    "The weather in Vienna is partly cloudy and 21°C."
                ],
            ),
            Golden(
                input=(
                    "Is Munich sunny today, "
                    "and what is the temperature?"
                ),
                context=[
                    "The weather in Munich is sunny and 22°C."
                ],
            ),
        ]
    )

    for golden in dataset.goldens:
        assistant = VoiceAssistant()

        result = assistant.ask(
            golden.input
        )

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=result["answer"],
            context=golden.context,
        )

        dataset.add_test_case(
            test_case
        )

    metric = HallucinationMetric(
        threshold=0.5,
        model=judge_model,
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[metric],
    )