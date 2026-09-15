from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import RoleAdherenceMetric
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import (
    end,
    proceed,
)
from deepeval.test_case import Turn

from src.agent.voice_agent import VoiceAssistant


CHATBOT_ROLE = (
    "A helpful and friendly voice assistant that provides "
    "concise and natural responses for spoken conversation."
)


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_role_adherence() -> None:
    assistants: dict[str, VoiceAssistant] = {}

    async def model_callback(
        input: str,
        thread_id: str,
    ) -> Turn:
        if thread_id not in assistants:
            assistants[thread_id] = VoiceAssistant()

        result = assistants[thread_id].ask(
            input
        )

        return Turn(
            role="assistant",
            content=result["answer"],
        )

    def ensure_multi_turn(
        simulated_user_turns: int,
    ):
        if simulated_user_turns < 3:
            return proceed()

        return end(
            reason=(
                "Enough turns generated for "
                "role adherence adapter."
            )
        )

    everyday_user = Persona(
        name="Everyday Voice Assistant User",
        characteristics=(
            "You interact naturally with a voice assistant. "
            "You ask one practical question at a time and wait "
            "for the assistant's response before continuing. "
            "You prefer clear, concise, and conversational responses."
        ),
    )

    style_challenging_user = Persona(
        name="Style-Challenging User",
        characteristics=(
            "You ask one question at a time. "
            "During the conversation, you sometimes pressure the assistant "
            "to communicate in an overly formal, technical, or impersonal "
            "way that is less suitable for spoken conversation. "
            "You wait for each response before continuing."
        ),
    )

    impatient_user = Persona(
        name="Impatient User",
        characteristics=(
            "You ask one question at a time but may become impatient, "
            "dismissive, or critical between requests. "
            "You still ask legitimate questions and wait for the "
            "assistant's answer before continuing."
        ),
    )

    everyday_assistance = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant answers, ask what attractions "
            "you should visit there. "
            "Then ask for Italian restaurant recommendations "
            "in Munich."
        ),
        expected_outcome=(
            "The assistant remains helpful, friendly, concise, "
            "and natural for spoken conversation throughout all "
            "three exchanges."
        ),
        persona=everyday_user,
    )

    conversational_style = ConversationalGolden(
        scenario=(
            "First ask what attractions you should visit in Vienna. "
            "After the assistant answers, ask it to describe one of "
            "those attractions using highly formal and technical language. "
            "Then ask a normal practical follow-up question about "
            "Italian restaurants in Vienna."
        ),
        expected_outcome=(
            "The assistant remains useful throughout the conversation "
            "while preserving a reasonably concise, natural, and "
            "spoken-conversation-friendly style even when the user "
            "pressures it toward an unsuitable communication style."
        ),
        persona=style_challenging_user,
    )

    friendly_under_pressure = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After receiving the answer, respond dismissively and ask "
            "about the weather in Vienna. "
            "Then briefly criticize the assistant and ask which of "
            "the two cities is warmer."
        ),
        expected_outcome=(
            "The assistant remains helpful, friendly, calm, concise, "
            "and constructive throughout the conversation despite "
            "the user's impatient or dismissive tone."
        ),
        persona=impatient_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            everyday_assistance,
            conversational_style,
            friendly_under_pressure,
        ],
        max_user_simulations=3,
    )

    assert len(test_cases) == 3

    assert all(
        test_case.turns
        for test_case in test_cases
    )

    assert all(
        len(
            [
                turn
                for turn in test_case.turns
                if turn.role == "user"
            ]
        ) >= 3
        for test_case in test_cases
    )

    for test_case in test_cases:
        test_case.chatbot_role = CHATBOT_ROLE

    for case_index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        print(
            f"\n=== CONVERSATION {case_index} ==="
        )

        for turn_index, turn in enumerate(
            test_case.turns,
            start=1,
        ):
            print(
                f"{turn_index}. "
                f"{turn.role}: "
                f"{turn.content}"
            )

    metric = RoleAdherenceMetric(
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )