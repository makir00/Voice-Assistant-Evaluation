import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from .prompts import SYSTEM_PROMPT
from .tools import (
    get_attractions,
    get_restaurants,
    get_weather,
)


load_dotenv()


class VoiceAssistant:
    """Multi-turn conversational assistant with tool-calling capabilities."""

    def __init__(self) -> None:
        model_name = os.getenv("OPENAI_MODEL")

        if not model_name:
            raise ValueError(
                "OPENAI_MODEL environment variable is not set."
            )

        self.messages: list[Any] = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ]

        self.model = ChatOpenAI(
            model=model_name,
            temperature=0,
            use_responses_api=True,
        ).bind_tools(
            [
                get_weather,
                get_attractions,
                get_restaurants,
            ]
        )

    def ask(
        self,
        message: str,
    ) -> dict[str, Any]:
        self.messages.append(
            HumanMessage(
                content=message
            )
        )

        all_tool_calls = []
        all_tool_results = []

        response = self._call_llm(
            self.messages,
            tool_choice="auto",
        )

        while response.tool_calls:
            self.messages.append(
                response
            )

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(
                    tool_call
                )

                all_tool_calls.append(
                    tool_call
                )

                all_tool_results.append(
                    {
                        "tool_name": tool_call["name"],
                        "result": tool_result,
                    }
                )

                self.messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

            response = self._call_llm(
                self.messages
            )

        self.messages.append(
            response
        )

        answer = self._extract_text(
            response.content
        )

        return {
            "answer": answer,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
        }

    def _call_llm(
        self,
        messages: list[Any],
        *,
        tool_choice: str | None = None,
    ) -> Any:
        if tool_choice is not None:
            return self.model.invoke(
                messages,
                tool_choice=tool_choice,
            )

        return self.model.invoke(
            messages
        )

    def _execute_tool(
        self,
        tool_call: dict[str, Any],
    ) -> str:
        if tool_call["name"] == "get_weather":
            return get_weather.invoke(
                tool_call["args"]
            )

        if tool_call["name"] == "get_attractions":
            return get_attractions.invoke(
                tool_call["args"]
            )

        if tool_call["name"] == "get_restaurants":
            return get_restaurants.invoke(
                tool_call["args"]
            )

        return (
            f"Unknown tool: "
            f"{tool_call['name']}"
        )

    @staticmethod
    def _extract_text(
        content: Any,
    ) -> str:
        if isinstance(
            content,
            str,
        ):
            return content

        if isinstance(
            content,
            list,
        ):
            text_parts = []

            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    text_parts.append(
                        item.get(
                            "text",
                            "",
                        )
                    )

            return "".join(
                text_parts
            )

        return str(
            content
        )

    def reset(
        self,
    ) -> None:
        self.messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ]


def main() -> None:
    assistant = VoiceAssistant()

    while True:
        message = input(
            "User: "
        )

        if message.lower() in {
            "exit",
            "quit",
        }:
            break

        result = assistant.ask(
            message
        )

        tool_names = ", ".join(
            tool_call["name"]
            for tool_call in result["tool_calls"]
        ) or "None"

        print(
            f"Tool calls: {tool_names}"
        )
        print(
            f"Agent: {result['answer']}"
        )


if __name__ == "__main__":
    main()