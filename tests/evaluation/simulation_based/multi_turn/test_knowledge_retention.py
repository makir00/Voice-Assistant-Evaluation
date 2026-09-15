from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import KnowledgeRetentionMetric
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


def test_knowledge_retention() -> None:
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
                "knowledge retention adapter."
            )
        )

    natural_follow_up_user = Persona(
        name="Natural Follow-up User",
        characteristics=(
            "You speak naturally and conversationally. "
            "You ask one question at a time and wait for the assistant's "
            "response before continuing. "
            "You use short follow-up questions and expect the assistant "
            "to remember information already established."
        ),
    )

    task_switching_user = Persona(
        name="Task-Switching User",
        characteristics=(
            "You ask one question at a time and may naturally move "
            "between different everyday topics. "
            "After changing topics, you sometimes return to information "
            "from an earlier part of the conversation."
        ),
    )

    concise_user = Persona(
        name="Concise User",
        characteristics=(
            "You ask short and direct questions one at a time. "
            "You wait for each response before continuing and avoid "
            "unnecessarily repeating information that has already "
            "been established."
        ),
    )

    weather_retention = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant responds, ask about the weather "
            "in Vienna. "
            "After receiving that response, ask which city is warmer "
            "without repeating the temperatures."
        ),
        expected_outcome=(
            "The assistant retains the previously provided weather "
            "information and correctly identifies Munich as warmer "
            "than Vienna."
        ),
        persona=natural_follow_up_user,
    )

    attraction_retention = ConversationalGolden(
        scenario=(
            "First ask what attractions you should visit in Munich. "
            "After the assistant responds, ask about the weather "
            "in Vienna. "
            "After receiving that response, return to the earlier "
            "Munich discussion and ask which attractions the assistant "
            "recommended there."
        ),
        expected_outcome=(
            "The assistant remembers the Munich attraction recommendations "
            "from earlier in the conversation despite the intervening "
            "Vienna weather question."
        ),
        persona=task_switching_user,
    )

    restaurant_context_retention = ConversationalGolden(
        scenario=(
            "First ask about the weather in Vienna. "
            "After the assistant responds, ask what attractions "
            "you should see there. "
            "After receiving that response, ask for Italian restaurant "
            "recommendations there without repeating the city name."
        ),
        expected_outcome=(
            "The assistant retains that the conversation is about Vienna "
            "and correctly uses that context when recommending Italian "
            "restaurants."
        ),
        persona=concise_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            weather_retention,
            attraction_retention,
            restaurant_context_retention,
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

    metric = KnowledgeRetentionMetric(
        threshold=0.6,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )