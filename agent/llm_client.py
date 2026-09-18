"""
OpenAI API Client.
Uses the official `openai` Python SDK with support for native Function Calling.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from openai import OpenAI, AuthenticationError, RateLimitError, APIError, APIConnectionError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: Any = None


class OpenAIClient:
    """Client using official OpenAI SDK for Chat Completions with tool calling."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            self._client = None
        else:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url and self.base_url != "https://api.openai.com/v1":
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """
        Sends chat completion request to OpenAI API via official SDK.
        """
        if not self.api_key or not self._client:
            raise ValueError(
                "OPENAI_API_KEY is not set!\n"
                "Please set it via environment variable or .env file:\n"
                "  export OPENAI_API_KEY='sk-...'\n"
                "or create a .env file with OPENAI_API_KEY=your_key"
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            completion = self._client.chat.completions.create(**kwargs)
            return self._parse_completion(completion)

        except AuthenticationError as e:
            raise PermissionError(
                f"Authentication Error (401): Invalid or missing OpenAI API Key.\nDetails: {e.message}"
            ) from e
        except RateLimitError as e:
            raise RuntimeError(
                f"Rate limit or quota exceeded (429) from OpenAI.\nDetails: {e.message}"
            ) from e
        except APIConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to OpenAI API: {str(e)}"
            ) from e
        except APIError as e:
            raise RuntimeError(
                f"OpenAI API error [{e.code}]: {e.message}"
            ) from e

    def _parse_completion(self, completion: Any) -> LLMResponse:
        """Parses OpenAI ChatCompletion object into structured LLMResponse."""
        choice = completion.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"
        content = message.content

        parsed_tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tc_id = tc.id
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                parsed_tool_calls.append(
                    ToolCall(id=tc_id, name=name, arguments=args, raw_arguments=raw_args)
                )

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            raw_response=completion,
        )
