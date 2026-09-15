from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    EvaluationDataset,
)
from deepeval.metrics import (
    ConversationCompletenessMetric,
    GoalAccuracyMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TopicAdherenceMetric,
    ToolUseMetric,
    TurnRelevancyMetric,
)
from deepeval.models import AnthropicModel
from deepeval.test_case import (
    Audio,
    ConversationalTestCase,
    ToolCall,
    Turn,
)

from src.voice.voice_pipeline import VoicePipeline


FIXTURES_DIR = Path(
    "audio_data/fixtures"
)

CHATBOT_ROLE = (
    "A helpful and friendly travel voice assistant that helps users "
    "with travel-related requests such as weather, attractions, "
    "restaurants, and practical travel recommendations, and politely "
    "declines requests outside the travel assistance domain."
)


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def run_fixture_conversation(
    fixture_name: str,
    *,
    scenario: str,
    expected_outcome: str,
    include_audio: bool = False,
    include_tool_outputs: bool = False,
) -> ConversationalTestCase:
    pipeline = VoicePipeline()

    fixture_dir = (
        FIXTURES_DIR / fixture_name
    )

    audio_turns = sorted(
        fixture_dir.glob("turn*.wav")
    )

    if not audio_turns:
        raise FileNotFoundError(
            f"No audio fixtures found in: {fixture_dir}"
        )

    turns = []

    for audio_path in audio_turns:
        result = pipeline.process_audio(
            str(audio_path)
        )

        print(
            f"\nUSER: {result['user_text']}"
        )
        print(
            f"ASSISTANT: {result['assistant_text']}"
        )

        turns.append(
            Turn(
                role="user",
                content=result["user_text"],
            )
        )

        assistant_tool_calls = []

        for tool_call, tool_result in zip(
            result["tool_calls"],
            result["tool_results"],
        ):
            tool_call_kwargs = {
                "name": tool_call["name"],
                "input_parameters": tool_call["args"],
            }

            if include_tool_outputs:
                tool_call_kwargs["output"] = (
                    tool_result["result"]
                )

            assistant_tool_calls.append(
                ToolCall(
                    **tool_call_kwargs
                )
            )

        assistant_turn_kwargs = {
            "role": "assistant",
            "content": result["assistant_text"],
            "tools_called": assistant_tool_calls,
        }

        if include_audio:
            assistant_turn_kwargs["audio"] = (
                Audio.from_bytes(
                    result["assistant_audio"],
                    mimeType="audio/wav",
                    sampleRate=24_000,
                )
            )

        turns.append(
            Turn(
                **assistant_turn_kwargs
            )
        )

    return ConversationalTestCase(
        scenario=scenario,
        expected_outcome=expected_outcome,
        chatbot_role=CHATBOT_ROLE,
        turns=turns,
    )


def test_main_voice_conversation() -> None:
    golden = ConversationalGolden(
        scenario=(
            "The user has a multi-turn spoken conversation with "
            "the assistant about weather, preparing for a walk, "
            "and related questions."
        ),
        expected_outcome=(
            "The assistant provides relevant, natural responses, "
            "uses appropriate tools, stays within the expected topics, "
            "and maintains its role throughout the conversation."
        ),
    )

    dataset = EvaluationDataset(
        goldens=[
            golden,
        ]
    )

    test_case = run_fixture_conversation(
        "main",
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        include_audio=True,
    )

    dataset.add_test_case(
        test_case
    )

    available_tools = [
        ToolCall(
            name="get_weather",
            description=(
                "Get the current weather for a city."
            ),
        ),
        ToolCall(
            name="get_attractions",
            description=(
                "Get popular attractions for a city."
            ),
        ),
        ToolCall(
            name="get_restaurants",
            description=(
                "Get restaurant recommendations "
                "for a city and cuisine."
            ),
        ),
    ]

    metrics = [
        ToolUseMetric(
            available_tools=available_tools,
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
        TurnRelevancyMetric(
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
        RoleAdherenceMetric(
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
        TopicAdherenceMetric(
            relevant_topics=[
                "weather",
                "preparing for a walk",
                "clothing and items needed for a walk",
            ],
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
    ]

    evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
    )


def test_knowledge_retention() -> None:
    golden = ConversationalGolden(
        scenario=(
            "The user provides information during a multi-turn "
            "spoken conversation and later refers back to it."
        ),
        expected_outcome=(
            "The assistant correctly remembers and uses information "
            "provided earlier in the conversation."
        ),
    )

    dataset = EvaluationDataset(
        goldens=[
            golden,
        ]
    )

    test_case = run_fixture_conversation(
        "knowledge_retention",
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
    )

    dataset.add_test_case(
        test_case
    )

    metric = KnowledgeRetentionMetric(
        threshold=0.5,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[metric],
    )


def test_conversation_completeness() -> None:
    golden = ConversationalGolden(
        scenario=(
            "The user expresses multiple intentions during a "
            "multi-turn spoken conversation."
        ),
        expected_outcome=(
            "The assistant addresses all relevant user intentions "
            "before the conversation ends."
        ),
    )

    dataset = EvaluationDataset(
        goldens=[
            golden,
        ]
    )

    test_case = run_fixture_conversation(
        "conversation_completeness",
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
    )

    dataset.add_test_case(
        test_case
    )

    metric = ConversationCompletenessMetric(
        threshold=0.5,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[metric],
    )


def test_goal_accuracy() -> None:
    golden = ConversationalGolden(
        scenario=(
            "The user attempts to accomplish a goal through a "
            "multi-turn spoken conversation with the assistant."
        ),
        expected_outcome=(
            "The assistant successfully helps the user accomplish "
            "the intended goal."
        ),
    )

    dataset = EvaluationDataset(
        goldens=[
            golden,
        ]
    )

    test_case = run_fixture_conversation(
        "goal_accuracy",
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        include_tool_outputs=True,
    )

    dataset.add_test_case(
        test_case
    )

    metric = GoalAccuracyMetric(
        threshold=0.5,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[metric],
    )