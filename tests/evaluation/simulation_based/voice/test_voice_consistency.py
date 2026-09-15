from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import VoiceConsistencyMetric
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import proceed
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def continue_until_max_turns():
    return proceed()


def test_voice_consistency(
    tmp_path: Path,
) -> None:
    concise_user = Persona(
        name="Concise Voice User",
        characteristics=(
            "You ask short and direct questions. "
            "You wait for the assistant to finish before continuing."
        ),
    )

    recommendation_user = Persona(
        name="Recommendation-Seeking Voice User",
        characteristics=(
            "You ask for recommendations and refine your request "
            "using natural follow-up questions."
        ),
    )

    planning_user = Persona(
        name="Step-by-Step Voice Planner",
        characteristics=(
            "You develop your request gradually across multiple turns "
            "and ask contextual follow-up questions."
        ),
    )

    conversational_goldens = [
        ConversationalGolden(
            scenario=(
                "First ask about the weather in Munich. "
                "After the assistant responds, ask about the weather "
                "in Vienna. "
                "Finally, ask which city is warmer."
            ),
            expected_outcome=(
                "The assistant maintains a consistent speaking voice "
                "throughout all responses."
            ),
            persona=concise_user,
        ),
        ConversationalGolden(
            scenario=(
                "First ask for Italian restaurant recommendations "
                "in Vienna. "
                "After receiving the recommendations, ask which one "
                "would be best for a casual dinner. "
                "Finally, ask which one would be best for a "
                "first-time visitor."
            ),
            expected_outcome=(
                "The assistant maintains a stable and consistent voice "
                "across the full conversation."
            ),
            persona=recommendation_user,
        ),
        ConversationalGolden(
            scenario=(
                "First ask about attractions to visit in Munich. "
                "After receiving the recommendations, ask which "
                "attraction is best for a first-time visitor. "
                "Finally, ask for Italian restaurant recommendations "
                "near that area."
            ),
            expected_outcome=(
                "The assistant preserves consistent vocal characteristics "
                "across short and longer spoken responses."
            ),
            persona=planning_user,
        ),
    ]

    output_dir = (
        tmp_path
        / "voice_consistency"
    )

    voice_config = VoiceConfig(
        connector=create_voice_connector(),
        output_dir=str(output_dir),
        combine_audio_files=True,
        record_call=True,
    )

    simulator = ConversationSimulator(
        voice_config=voice_config,
        stopping_controller=continue_until_max_turns,
    )

    test_cases = simulator.simulate(
        conversational_goldens=conversational_goldens,
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

    metric = VoiceConsistencyMetric(
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )