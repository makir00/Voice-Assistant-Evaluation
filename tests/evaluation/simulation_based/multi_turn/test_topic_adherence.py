from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import TopicAdherenceMetric
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import (
    end,
    proceed,
)
from deepeval.test_case import Turn

from src.agent.voice_agent import VoiceAssistant


RELEVANT_TOPICS = [
    "weather information",
    "travel attractions and sightseeing",
    "restaurant recommendations",
]


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_topic_adherence() -> None:
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
                "topic adherence adapter."
            )
        )

    focused_traveler = Persona(
        name="Focused Traveler",
        characteristics=(
            "You ask one practical travel-related question at a time. "
            "You wait for the assistant's response before continuing. "
            "You prefer concise and conversational responses."
        ),
    )

    topic_switching_user = Persona(
        name="Topic-Switching User",
        characteristics=(
            "You ask one question at a time and sometimes switch "
            "from travel-related questions to clearly unrelated topics. "
            "You wait for each response before continuing and may later "
            "return to the original travel discussion."
        ),
    )

    mixed_scope_user = Persona(
        name="Mixed-Scope User",
        characteristics=(
            "You ask one request at a time. "
            "You naturally alternate between supported travel questions "
            "and requests that fall outside the assistant's intended topics."
        ),
    )

    in_scope_conversation = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask about attractions "
            "to visit in Munich. "
            "After receiving that response, ask for restaurant "
            "recommendations there."
        ),
        expected_outcome=(
            "The assistant answers all three questions because they "
            "remain within the supported topics of weather information, "
            "travel attractions, and restaurant recommendations."
        ),
        persona=focused_traveler,
    )

    explicit_topic_switch = ConversationalGolden(
        scenario=(
            "First ask about attractions in Vienna. "
            "After the assistant responds, switch topics and ask "
            "how to implement a sorting algorithm in Python. "
            "After receiving that response, return to the travel "
            "discussion and ask for restaurant recommendations "
            "in Vienna."
        ),
        expected_outcome=(
            "The assistant answers the supported travel questions, "
            "avoids substantively answering the unrelated programming "
            "request, and resumes helping when the conversation returns "
            "to a supported topic."
        ),
        persona=topic_switching_user,
    )

    mixed_scope_conversation = ConversationalGolden(
        scenario=(
            "First ask about the weather in Berlin. "
            "After the assistant responds, ask it to write "
            "a short poem about cats. "
            "After receiving that response, return to the travel topic "
            "and ask about attractions to visit in Berlin."
        ),
        expected_outcome=(
            "The assistant answers the supported weather and attraction "
            "questions while avoiding a substantive response to the "
            "unrelated creative-writing request."
        ),
        persona=mixed_scope_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            in_scope_conversation,
            explicit_topic_switch,
            mixed_scope_conversation,
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

    metric = TopicAdherenceMetric(
        relevant_topics=RELEVANT_TOPICS,
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )