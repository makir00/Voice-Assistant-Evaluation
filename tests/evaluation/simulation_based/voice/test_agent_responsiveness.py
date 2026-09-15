from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import AgentResponsivenessMetric
from deepeval.simulator import ConversationSimulator
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def test_agent_responsiveness(
    tmp_path: Path,
) -> None:
    direct_user = Persona(
        name="Direct Voice Caller",
        characteristics=(
            "You speak clearly and naturally. "
            "You ask one short question at a time. "
            "You wait for the assistant's response before continuing "
            "and do not repeat yourself unless necessary."
        ),
    )

    contextual_user = Persona(
        name="Contextual Voice Caller",
        characteristics=(
            "You speak naturally and use short follow-up questions. "
            "You ask one request per turn and expect the assistant "
            "to remember information established earlier."
        ),
    )

    task_switching_user = Persona(
        name="Task-Switching Voice Caller",
        characteristics=(
            "You ask one request at a time and wait for each response. "
            "You naturally switch between different supported travel tasks "
            "during the conversation."
        ),
    )

    direct_questions = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask about the weather "
            "in Vienna. "
            "After receiving that response, ask which city is warmer."
        ),
        expected_outcome=(
            "The assistant responds to every request without requiring "
            "the user to repeat or rephrase any question."
        ),
        persona=direct_user,
    )

    contextual_follow_up = ConversationalGolden(
        scenario=(
            "First ask what attractions you should visit in Vienna. "
            "After the assistant responds, ask for Italian restaurant "
            "recommendations there without repeating the city name. "
            "After receiving that response, ask what the weather "
            "is like there."
        ),
        expected_outcome=(
            "The assistant responds to every request and handles "
            "contextual follow-ups without requiring the user "
            "to repeat previously established information."
        ),
        persona=contextual_user,
    )

    task_switching = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After receiving the response, ask which attractions "
            "you should visit there. "
            "After the assistant responds, switch to restaurant "
            "recommendations and ask for Italian restaurants in Munich."
        ),
        expected_outcome=(
            "The assistant responds successfully to every turn while "
            "the user switches between supported travel tasks and does "
            "not require a user reprompt."
        ),
        persona=task_switching_user,
    )

    voice_config = VoiceConfig(
        connector=create_voice_connector(),
        output_dir=str(
            tmp_path / "agent_responsiveness"
        ),
        combine_audio_files=True,
    )

    simulator = ConversationSimulator(
        voice_config=voice_config,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            direct_questions,
            contextual_follow_up,
            task_switching,
        ],
        max_user_simulations=3,
    )

    assert len(test_cases) == 3

    for test_case in test_cases:
        assert test_case.turns

        user_turns = [
            turn
            for turn in test_case.turns
            if turn.role == "user"
        ]

        assistant_turns = [
            turn
            for turn in test_case.turns
            if turn.role == "assistant"
        ]

        assert len(user_turns) == 3
        assert len(assistant_turns) == 3

        assert all(
            turn.audio is not None
            for turn in assistant_turns
        )

    metric = AgentResponsivenessMetric(
        threshold=0.8,
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )