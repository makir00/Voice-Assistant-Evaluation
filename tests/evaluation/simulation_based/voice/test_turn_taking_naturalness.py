from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import TurnTakingNaturalnessMetric
from deepeval.simulator import (
    ConversationSimulator,
    SimulationNode,
)
from deepeval.simulator.controller import proceed
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def continue_until_max_turns():
    return proceed()


def build_fixed_three_turn_graph(
    user_messages: list[str],
) -> SimulationNode:
    def next_user_turn(turns):
        user_turn_count = sum(
            1
            for turn in turns
            if turn.role == "user"
        )

        return user_messages[user_turn_count]

    return SimulationNode(
        action=next_user_turn,
        name="fixed_three_turn_conversation",
        max_visits=3,
    )


def test_turn_taking_naturalness(
    tmp_path: Path,
) -> None:
    short_question_user = Persona(
        name="Short-Question Voice Caller",
        characteristics=(
            "You speak naturally using short and direct questions. "
            "You wait for the assistant to finish before continuing."
        ),
    )

    conversational_user = Persona(
        name="Conversational Voice Caller",
        characteristics=(
            "You speak at a natural conversational pace. "
            "You use short follow-up questions and wait for the assistant "
            "to finish before taking your next turn."
        ),
    )

    planning_user = Persona(
        name="Step-by-Step Voice Planner",
        characteristics=(
            "You develop your request gradually across several turns. "
            "You allow the assistant to finish each response before "
            "asking the next related question."
        ),
    )

    conversations = [
        {
            "golden": ConversationalGolden(
                scenario=(
                    "Compare the weather in Munich and Vienna."
                ),
                expected_outcome=(
                    "The conversation has smooth hand-offs between "
                    "the user and assistant."
                ),
                persona=short_question_user,
            ),
            "messages": [
                "What's the weather like in Munich?",
                "What's the weather like in Vienna?",
                "Which city is warmer?",
            ],
            "output_dir": "weather_comparison",
        },
        {
            "golden": ConversationalGolden(
                scenario=(
                    "Ask for Italian restaurant recommendations "
                    "in Vienna and refine the choice."
                ),
                expected_outcome=(
                    "The user and assistant take turns naturally "
                    "without awkward delays or interruptions."
                ),
                persona=conversational_user,
            ),
            "messages": [
                "Could you recommend some Italian restaurants in Vienna?",
                "Which one would be best for a casual dinner?",
                "Which one would you recommend for a first-time visitor?",
            ],
            "output_dir": "vienna_restaurants",
        },
        {
            "golden": ConversationalGolden(
                scenario=(
                    "Plan a short first-time visit to Munich."
                ),
                expected_outcome=(
                    "The conversation maintains natural turn-taking "
                    "while the user develops the request step by step."
                ),
                persona=planning_user,
            ),
            "messages": [
                "What attractions should I visit in Munich?",
                "Which one would be best for a first-time visitor?",
                "Can you recommend Italian restaurants near Marienplatz?",
            ],
            "output_dir": "munich_planning",
        },
    ]

    test_cases = []

    for conversation in conversations:
        simulation_graph = build_fixed_three_turn_graph(
            conversation["messages"]
        )

        voice_config = VoiceConfig(
            connector=create_voice_connector(),
            output_dir=str(
                tmp_path
                / "turn_taking_naturalness"
                / conversation["output_dir"]
            ),
            combine_audio_files=True,
            record_call=True,
        )

        simulator = ConversationSimulator(
            voice_config=voice_config,
            simulation_graph=simulation_graph,
            stopping_controller=continue_until_max_turns,
        )

        generated_cases = simulator.simulate(
            conversational_goldens=[
                conversation["golden"],
            ],
            max_user_simulations=3,
        )

        assert len(generated_cases) == 1

        test_cases.extend(
            generated_cases
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
            and turn.audio.start_time is not None
            for turn in test_case.turns
        )

    metric = TurnTakingNaturalnessMetric(
        threshold=0.6,
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )