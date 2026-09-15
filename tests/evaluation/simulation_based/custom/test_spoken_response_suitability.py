from pathlib import Path

from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import ConversationalGEval
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import MultiTurnParams
from deepeval.voice import VoiceConfig

from src.adapter.voice_connector import create_voice_connector


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_spoken_response_suitability(
    tmp_path: Path,
) -> None:
    voice_user = Persona(
        name="Voice Assistant User",
        characteristics=(
            "You interact with the assistant as if you were speaking "
            "rather than typing. "
            "You prefer short, natural, easy-to-follow responses "
            "that are comfortable to hear aloud."
        ),
    )

    spoken_conversation = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask whether you should bring "
            "a jacket for a walk. "
            "After receiving that response, ask about attractions "
            "to visit in Munich. "
            "Finally, ask for Italian restaurant recommendations "
            "in Munich."
        ),
        expected_outcome=(
            "The assistant provides concise, natural, and easy-to-follow "
            "spoken responses throughout the conversation. "
            "Responses should be suitable for voice interaction and "
            "avoid unnecessary verbosity."
        ),
        persona=voice_user,
    )

    voice_config = VoiceConfig(
        connector=create_voice_connector(),
        output_dir=str(
            tmp_path / "spoken_response_suitability"
        ),
        combine_audio_files=True,
    )

    simulator = ConversationSimulator(
        voice_config=voice_config,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            spoken_conversation,
        ],
        max_user_simulations=4,
    )

    assert len(test_cases) == 1
    assert test_cases[0].turns

    assistant_turns = [
        turn
        for turn in test_cases[0].turns
        if turn.role == "assistant"
    ]

    assert assistant_turns

    assert all(
        turn.audio is not None
        for turn in assistant_turns
    )

    metric = ConversationalGEval(
        name="Spoken Response Suitability",
        criteria=(
            "Evaluate whether the assistant's responses are well suited "
            "for spoken voice interaction. "
            "Responses should be concise, natural, conversational, "
            "easy to understand when heard aloud, and free from "
            "unnecessary verbosity or awkwardly written phrasing."
        ),
        evaluation_params=[
            MultiTurnParams.CONTENT,
        ],
        threshold=0.75,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )