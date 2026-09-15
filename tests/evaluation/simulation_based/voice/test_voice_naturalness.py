from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import VoiceNaturalnessMetric
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import proceed
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def continue_until_max_turns():
    return proceed()


def test_voice_naturalness(
    tmp_path: Path,
) -> None:
    concise_user = Persona(
        name="Concise Voice User",
        characteristics=(
            "You speak clearly and naturally. "
            "You ask one short question at a time. "
            "You wait for the assistant's response before continuing."
        ),
    )

    recommendation_user = Persona(
        name="Recommendation-Seeking Voice User",
        characteristics=(
            "You speak in a relaxed conversational manner. "
            "You ask one recommendation question at a time. "
            "You wait for the assistant's response before continuing "
            "with a related follow-up."
        ),
    )

    mixed_length_user = Persona(
        name="Mixed-Length Voice User",
        characteristics=(
            "You speak naturally and conversationally. "
            "You ask one request per turn. "
            "You alternate between short factual questions and questions "
            "that require somewhat longer answers."
        ),
    )

    short_responses = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask about the weather "
            "in Vienna. "
            "After receiving that response, ask which city is warmer."
        ),
        expected_outcome=(
            "The assistant produces natural-sounding spoken responses "
            "throughout the conversation."
        ),
        persona=concise_user,
    )

    recommendation_responses = ConversationalGolden(
        scenario=(
            "First ask what attractions are worth visiting in Vienna. "
            "After the assistant responds, ask which attraction "
            "would be best for a first-time visitor. "
            "After receiving that response, ask for Italian restaurant "
            "recommendations in Vienna."
        ),
        expected_outcome=(
            "The assistant maintains natural spoken delivery across "
            "recommendation and follow-up responses."
        ),
        persona=recommendation_user,
    )

    mixed_response_lengths = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask what attractions "
            "are worth visiting there. "
            "After receiving that response, ask which attraction "
            "would be the best choice if you only had one hour."
        ),
        expected_outcome=(
            "The assistant maintains natural-sounding speech across "
            "both short and longer spoken responses."
        ),
        persona=mixed_length_user,
    )

    voice_config = VoiceConfig(
        connector=create_voice_connector(),
        output_dir=str(
            tmp_path / "voice_naturalness"
        ),
        combine_audio_files=True,
    )

    simulator = ConversationSimulator(
        voice_config=voice_config,
        stopping_controller=continue_until_max_turns,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            short_responses,
            recommendation_responses,
            mixed_response_lengths,
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

    metric = VoiceNaturalnessMetric(
        threshold=0.7,
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )