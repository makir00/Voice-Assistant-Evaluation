from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import GoalAccuracyMetric
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import (
    end,
    proceed,
)
from deepeval.test_case import ToolCall, Turn

from src.agent.voice_agent import VoiceAssistant


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_goal_accuracy() -> None:
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

        tools_called = []

        for tool_call, tool_result in zip(
            result["tool_calls"],
            result["tool_results"],
        ):
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call.get(
                        "args",
                    ),
                    output=tool_result["result"],
                )
            )

        return Turn(
            role="assistant",
            content=result["answer"],
            tools_called=tools_called or None,
        )

    def ensure_multi_turn(
        simulated_user_turns: int,
    ):
        if simulated_user_turns < 3:
            return proceed()

        return end(
            reason=(
                "Enough turns generated for "
                "goal accuracy adapter."
            )
        )

    comparison_user = Persona(
        name="City Comparison User",
        characteristics=(
            "You ask one question at a time because you want to compare "
            "different options before making a decision. "
            "You wait for the assistant's answer before asking "
            "a short follow-up based on information already established."
        ),
    )

    trip_planning_user = Persona(
        name="Trip Planning User",
        characteristics=(
            "You are planning a short city trip and collect information "
            "step by step. "
            "You ask one practical question at a time and wait for "
            "the assistant before continuing with your next need."
        ),
    )

    recommendation_user = Persona(
        name="Recommendation-Seeking User",
        characteristics=(
            "You ask one recommendation question at a time. "
            "You naturally continue with short follow-ups and expect "
            "the assistant to preserve information established earlier "
            "in the conversation."
        ),
    )

    compare_weather = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant answers, ask about the weather in Vienna. "
            "Then ask which of the two cities is warmer based on "
            "the information already provided."
        ),
        expected_outcome=(
            "The assistant provides the weather for both cities "
            "and correctly determines that Munich is warmer than Vienna."
        ),
        persona=comparison_user,
    )

    plan_vienna_trip = ConversationalGolden(
        scenario=(
            "You are planning a trip to Vienna. "
            "First ask about the weather there. "
            "After the assistant answers, ask what attractions "
            "are worth visiting in Vienna. "
            "Then ask for Italian restaurant recommendations "
            "in the same city."
        ),
        expected_outcome=(
            "The assistant helps the user plan the Vienna trip by providing "
            "relevant weather information, attraction recommendations, "
            "and Italian restaurant recommendations."
        ),
        persona=trip_planning_user,
    )

    contextual_recommendation = ConversationalGolden(
        scenario=(
            "First ask for attractions to visit in Munich. "
            "After the assistant answers, ask which recommendation "
            "would be especially suitable for a first-time visitor. "
            "Then ask for Italian restaurant recommendations there "
            "without repeating the city name."
        ),
        expected_outcome=(
            "The assistant helps the user explore Munich, responds "
            "appropriately to the attraction follow-up, and provides "
            "Italian restaurant recommendations while maintaining "
            "the established Munich context."
        ),
        persona=recommendation_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            compare_weather,
            plan_vienna_trip,
            contextual_recommendation,
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

            if turn.tools_called:
                for tool in turn.tools_called:
                    print(
                        f"   Tool: {tool.name}"
                    )
                    print(
                        f"   Input: {tool.input_parameters}"
                    )
                    print(
                        f"   Output: {tool.output}"
                    )

    metric = GoalAccuracyMetric(
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )