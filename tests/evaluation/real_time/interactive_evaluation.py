# Runs a real-time voice conversation and evaluates the resulting conversation trace with DeepEval metrics.

from deepeval import evaluate
from deepeval.metrics import (
    RoleAdherenceMetric,
    ToolUseMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import (
    ConversationalTestCase,
    ToolCall,
    Turn,
)

from src.voice.interactive_session import run_interactive_session

CHATBOT_ROLE = (
    "A helpful and friendly travel voice assistant that helps users "
    "with travel-related requests such as weather, attractions, "
    "restaurants, and practical travel recommendations, and politely "
    "declines requests outside the travel assistance domain."
)


def build_test_case(
    conversation_trace: list[dict],
) -> ConversationalTestCase:
    turns = []

    for result in conversation_trace:
        turns.append(
            Turn(
                role="user",
                content=result["user_text"],
            )
        )

        tool_calls = [
            ToolCall(
                name=tool_call["name"],
                input_parameters=tool_call["args"],
            )
            for tool_call in result["tool_calls"]
        ]

        turns.append(
            Turn(
                role="assistant",
                content=result["assistant_text"],
                tools_called=tool_calls,
            )
        )

    return ConversationalTestCase(
        chatbot_role=CHATBOT_ROLE,
        turns=turns,
    )


def main() -> None:
    conversation_trace = run_interactive_session()

    if not conversation_trace:
        print("No conversation was recorded.")
        return

    test_case = build_test_case(
        conversation_trace
    )

    available_tools = [
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
                "Get restaurant recommendations "
                "for a city and cuisine."
            ),
        ),
    ]

    metrics = [
        ToolUseMetric(
            available_tools=available_tools,
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
        TurnRelevancyMetric(
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
        RoleAdherenceMetric(
            threshold=0.75,
            include_reason=True,
            verbose_mode=True,
        ),
    ]

    evaluate(
        test_cases=[test_case],
        metrics=metrics,
    )


if __name__ == "__main__":
    main()