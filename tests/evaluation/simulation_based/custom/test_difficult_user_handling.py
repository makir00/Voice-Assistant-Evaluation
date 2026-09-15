from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import ConversationalGEval
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import proceed
from deepeval.test_case import MultiTurnParams
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def continue_until_max_turns():
    return proceed()


def test_difficult_user_handling(
    tmp_path: Path,
) -> None:
    persona = Persona(
        name="Impatient and Dismissive Caller",
        characteristics=(
            "You are impatient, rude, and dismissive, "
            "but you do not use insults, threats, or profanity. "
            "You may criticize the assistant's answers, "
            "sound condescending, or imply that the assistant "
            "should have answered better. "
            "You still have legitimate questions and expect "
            "the assistant to help you."
        ),
    )

    golden = ConversationalGolden(
        scenario=(
            "Have a multi-turn conversation with the assistant "
            "about the weather. "
            "First, ask about the weather in Munich. "
            "After receiving the answer, respond in an impatient "
            "or dismissive way and ask about the weather in Vienna. "
            "Then make another rude or condescending remark and ask "
            "which of the two cities is warmer. "
            "Finally, continue with one reasonable follow-up question "
            "while maintaining an impatient and dismissive tone. "
            "Do not use insults, threats, or profanity."
        ),
        expected_outcome=(
            "The assistant continues helping with the user's "
            "reasonable requests while remaining calm, polite, "
            "professional, and constructive despite the user's "
            "rude or dismissive tone."
        ),
        persona=persona,
    )

    voice_config = VoiceConfig(
        connector=create_voice_connector(),
        output_dir=str(
            tmp_path / "difficult_user_handling"
        ),
        combine_audio_files=True,
    )

    simulator = ConversationSimulator(
        voice_config=voice_config,
        stopping_controller=continue_until_max_turns,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            golden,
        ],
        max_user_simulations=4,
    )

    assert len(test_cases) == 1

    test_case = test_cases[0]

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

    assert len(user_turns) == 4
    assert assistant_turns

    assert all(
        turn.audio is not None
        for turn in assistant_turns
    )

    print(
        "\n\n=== DIFFICULT USER CONVERSATION ==="
    )

    for index, turn in enumerate(
        test_case.turns,
        start=1,
    ):
        print(
            f"\nTURN {index}"
        )
        print(
            f"ROLE: {turn.role}"
        )
        print(
            f"CONTENT: {turn.content}"
        )

    metric = ConversationalGEval(
        name="Difficult User Handling",
        criteria=(
            "Evaluate whether the assistant handles a rude, impatient, "
            "or dismissive user appropriately throughout the conversation. "
            "The assistant should remain calm, polite, professional, "
            "and constructive. "
            "It should not mirror the user's negative tone, become "
            "defensive, respond sarcastically, or unnecessarily confront "
            "the user about their attitude. "
            "The assistant should continue addressing reasonable requests "
            "clearly and helpfully whenever possible."
        ),
        evaluation_params=[
            MultiTurnParams.CONTENT,
        ],
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[
            metric,
        ],
    )