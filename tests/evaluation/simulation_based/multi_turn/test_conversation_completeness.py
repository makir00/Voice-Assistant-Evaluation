from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import ConversationCompletenessMetric
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


def test_conversation_completeness() -> None:
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
                "conversation completeness adapter."
            )
        )

    multi_need_user = Persona(
        name="Multi-Need Traveler",
        characteristics=(
            "You ask one travel question at a time. "
            "You wait for the assistant's answer before introducing "
            "your next related need. "
            "You expect every request raised during the conversation "
            "to be addressed."
        ),
    )

    planning_user = Persona(
        name="Step-by-Step Trip Planner",
        characteristics=(
            "You build your travel request gradually across multiple turns. "
            "You ask one question at a time and wait for the assistant "
            "before introducing another practical need related to "
            "the same trip."
        ),
    )

    detail_checking_user = Persona(
        name="Detail-Checking User",
        characteristics=(
            "You ask one request at a time and pay attention to whether "
            "each need has been answered. "
            "You may later return to an earlier part of the conversation "
            "to continue or verify that request."
        ),
    )

    combined_travel_needs = ConversationalGolden(
        scenario=(
            "You are planning time in Munich. "
            "First ask about the weather there. "
            "After the assistant answers, ask what attractions "
            "you should visit in Munich. "
            "Then ask for Italian restaurant recommendations "
            "in the same city."
        ),
        expected_outcome=(
            "The assistant addresses all three user intentions raised "
            "throughout the conversation: Munich weather information, "
            "Munich attraction recommendations, and Italian restaurant "
            "recommendations in Munich."
        ),
        persona=multi_need_user,
    )

    progressive_trip_planning = ConversationalGolden(
        scenario=(
            "You are planning a trip to Vienna. "
            "First ask about the weather there. "
            "After the assistant answers, ask which attractions "
            "you should visit in Vienna. "
            "Then ask for Italian restaurant recommendations there "
            "without unnecessarily repeating the city name."
        ),
        expected_outcome=(
            "The assistant satisfies each user intention introduced "
            "throughout the conversation: Vienna weather information, "
            "Vienna attraction recommendations, and Italian restaurant "
            "recommendations in Vienna."
        ),
        persona=planning_user,
    )

    return_to_earlier_need = ConversationalGolden(
        scenario=(
            "First ask what attractions you should visit in Munich. "
            "After the assistant answers, ask about the weather in Vienna. "
            "Then return to the earlier Munich context and ask for "
            "Italian restaurant recommendations there."
        ),
        expected_outcome=(
            "The assistant satisfies all user needs raised throughout "
            "the conversation, including Munich attraction recommendations, "
            "Vienna weather information, and Italian restaurant "
            "recommendations in Munich when the user returns "
            "to the earlier city context."
        ),
        persona=detail_checking_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            combined_travel_needs,
            progressive_trip_planning,
            return_to_earlier_need,
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

    metric = ConversationCompletenessMetric(
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )