from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import TurnRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import (
    end,
    proceed,
)
from deepeval.test_case import Turn

from src.agent.voice_agent import VoiceAssistant


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_turn_relevancy() -> None:
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
                "turn relevancy adapter."
            )
        )

    focused_user = Persona(
        name="Focused Traveler",
        characteristics=(
            "You ask one concise travel-related question at a time. "
            "You wait for the assistant's response before continuing. "
            "You naturally follow up on information provided earlier "
            "in the conversation."
        ),
    )

    context_switching_user = Persona(
        name="Context-Switching Traveler",
        characteristics=(
            "You ask one question at a time and sometimes change "
            "the subject during a conversation. "
            "You wait for each response and expect the assistant "
            "to focus on your latest request instead of unnecessarily "
            "returning to the previous topic."
        ),
    )

    contextual_user = Persona(
        name="Contextual Follow-Up User",
        characteristics=(
            "You ask one question at a time and frequently use short "
            "follow-ups with references such as 'there', 'those', "
            "and 'that city'. "
            "You expect the assistant to interpret them using "
            "the existing conversation context."
        ),
    )

    munich_planning = ConversationalGolden(
        scenario=(
            "First ask about attractions in Munich. "
            "After the assistant responds, ask which of those attractions "
            "would be best for a first-time visitor. "
            "After receiving that response, ask for Italian restaurant "
            "recommendations in Munich."
        ),
        expected_outcome=(
            "The assistant stays focused on each current request "
            "and provides relevant Munich attraction and restaurant "
            "recommendations throughout the conversation."
        ),
        persona=focused_user,
    )

    topic_switch = ConversationalGolden(
        scenario=(
            "First ask about the weather in Berlin. "
            "After the assistant responds, change the subject and ask "
            "for attractions in Vienna. "
            "After receiving that response, ask which of those Vienna "
            "attractions would be most suitable for a first-time visitor."
        ),
        expected_outcome=(
            "The assistant responds to each current request and correctly "
            "adapts when the user changes the subject from Berlin weather "
            "to Vienna attractions."
        ),
        persona=context_switching_user,
    )

    contextual_follow_up = ConversationalGolden(
        scenario=(
            "First ask for Italian restaurant recommendations in Vienna. "
            "After the assistant responds, ask which of those restaurants "
            "would be best for a casual dinner. "
            "After receiving that response, ask what the weather is like "
            "there without repeating the city name."
        ),
        expected_outcome=(
            "The assistant correctly follows the conversational context, "
            "responds to the restaurant follow-up, and interprets "
            "'there' as Vienna when answering the weather question."
        ),
        persona=contextual_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            munich_planning,
            topic_switch,
            contextual_follow_up,
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

    metric = TurnRelevancyMetric(
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )