from deepeval import evaluate
from deepeval.dataset import (
    ConversationalGolden,
    Persona,
)
from deepeval.metrics import ToolUseMetric
from deepeval.models import AnthropicModel
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import (
    end,
    proceed,
)
from deepeval.test_case import ToolCall, Turn

from src.agent.voice_agent import VoiceAssistant


AVAILABLE_TOOLS = [
    ToolCall(
        name="get_weather",
        description="Get the current weather for a city.",
    ),
    ToolCall(
        name="get_attractions",
        description="Get popular attractions for a city.",
    ),
    ToolCall(
        name="get_restaurants",
        description=(
            "Get restaurant recommendations for a city and cuisine."
        ),
    ),
]


def create_judge_model() -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-4-6",
    )


def test_tool_use() -> None:
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

        tools_called = [
            ToolCall(
                name=tool_call["name"],
                input_parameters=tool_call.get(
                    "args",
                ),
            )
            for tool_call in result["tool_calls"]
        ]

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
            reason="Enough turns generated for multi-turn tool adapter."
        )

    tool_switching_user = Persona(
        name="Tool-Switching Travel User",
        characteristics=(
            "You ask one travel question at a time. "
            "You wait for the assistant's answer before asking "
            "a follow-up about another travel need. "
            "You naturally rely on previously established city context."
        ),
    )

    contextual_user = Persona(
        name="Contextual Travel User",
        characteristics=(
            "You ask short follow-up questions and naturally use "
            "references such as 'there' instead of repeating a city "
            "that has already been established in the conversation."
        ),
    )

    multi_need_user = Persona(
        name="Multi-Need Travel User",
        characteristics=(
            "You use the assistant for several practical travel needs "
            "within the same conversation. "
            "You ask each need separately and naturally move between "
            "weather, sightseeing, and restaurant questions."
        ),
    )

    contextual_tool_switching = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant answers, ask about the weather "
            "in Vienna without repeating the full original question. "
            "Then ask what attractions you should visit there, "
            "using the established Vienna context."
        ),
        expected_outcome=(
            "The assistant uses the weather tool for Munich and Vienna, "
            "then correctly resolves 'there' as Vienna and uses "
            "the attractions tool for the final request."
        ),
        persona=tool_switching_user,
    )

    contextual_arguments_and_switching = ConversationalGolden(
        scenario=(
            "First ask what attractions you should visit in Vienna. "
            "After the assistant answers, ask for Italian restaurant "
            "recommendations there without repeating the city name. "
            "Then ask what the weather is like there."
        ),
        expected_outcome=(
            "The assistant uses the attractions tool for Vienna, "
            "correctly resolves 'there' as Vienna when using the "
            "restaurant tool with Italian cuisine, and then uses "
            "the weather tool for Vienna."
        ),
        persona=contextual_user,
    )

    multiple_tool_selection = ConversationalGolden(
        scenario=(
            "First ask about the weather in Munich. "
            "After the assistant answers, ask what attractions "
            "you should visit there. "
            "Then ask for German restaurant recommendations "
            "in the same city."
        ),
        expected_outcome=(
            "The assistant selects the weather, attractions, and "
            "restaurant tools at the appropriate points in the "
            "conversation and provides the correct city and "
            "cuisine arguments."
        ),
        persona=multi_need_user,
    )

    simulator = ConversationSimulator(
        model_callback=model_callback,
        stopping_controller=ensure_multi_turn,
    )

    test_cases = simulator.simulate(
        conversational_goldens=[
            contextual_tool_switching,
            contextual_arguments_and_switching,
            multiple_tool_selection,
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

    metric = ToolUseMetric(
        available_tools=AVAILABLE_TOOLS,
        threshold=0.7,
        model=create_judge_model(),
        include_reason=True,
        verbose_mode=True,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[metric],
    )