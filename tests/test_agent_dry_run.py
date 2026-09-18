import unittest
import os
from agent.config import Config, load_config
from agent.llm_client import OpenAIClient
from agent.core import Agent


class TestAgentDryRun(unittest.TestCase):
    def test_config_defaults(self):
        cfg = load_config()
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")

    def test_config_override(self):
        cfg = load_config(model_override="gpt-4o")
        self.assertEqual(cfg.model, "gpt-4o")

    def test_missing_api_key_raises_informative_error(self):
        # With empty API key, client.chat should give clear guidance
        client = OpenAIClient(api_key="", model="gpt-4o-mini")
        with self.assertRaises(ValueError) as ctx:
            client.chat(messages=[{"role": "user", "content": "hello"}])
        self.assertIn("OPENAI_API_KEY is not set", str(ctx.exception))

    def test_agent_initialization(self):
        agent = Agent(model="gpt-4o-mini")
        self.assertEqual(agent.config.model, "gpt-4o-mini")
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertIn("HUMAN-IN-THE-LOOP VERIFICATION", agent.system_prompt)

    def test_agent_with_confirmation_callback(self):
        cb_called = []

        def my_cb(tool_name, args):
            cb_called.append((tool_name, args))
            return True

        agent = Agent(model="gpt-4o-mini", confirmation_callback=my_cb)
        self.assertIsNotNone(agent.confirmation_callback)

    def test_confirmation_callback_declined(self):
        from agent.llm_client import LLMResponse, ToolCall

        agent = Agent(
            model="gpt-4o-mini",
            confirmation_callback=lambda name, args: False,
            enable_dag=False,
        )

        class MockClientReject:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="book_flight",
                                arguments={"flight_number": "BA 178"},
                                raw_arguments='{"flight_number": "BA 178"}',
                            )
                        ],
                    )
                return LLMResponse(content="Booking was cancelled as requested.")

        agent.client = MockClientReject()
        ans = agent.run("book it", max_steps=2)
        tool_obs = [m for m in agent.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_obs), 1)
        self.assertIn("Action cancelled: Human verification declined", tool_obs[0]["content"])
        self.assertEqual(ans, "Booking was cancelled as requested.")

    def test_confirmation_callback_approved(self):
        from agent.llm_client import LLMResponse, ToolCall

        agent = Agent(
            model="gpt-4o-mini",
            confirmation_callback=lambda name, args: True,
            enable_dag=False,
        )

        class MockClientApprove:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_2",
                                name="book_flight",
                                arguments={
                                    "flight_number": "BA 178",
                                    "passenger_name": "Alice Smith",
                                    "date": "2026-09-25",
                                },
                                raw_arguments="{}",
                            )
                        ],
                    )
                return LLMResponse(content="Flight booked successfully!")

        agent.client = MockClientApprove()
        ans = agent.run("yes book", max_steps=2)
        tool_obs = [m for m in agent.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_obs), 1)
        self.assertIn("CONFIRMED", tool_obs[0]["content"])
        self.assertIn("BK-BA178-", tool_obs[0]["content"])
        self.assertEqual(ans, "Flight booked successfully!")

    def test_agent_dag_blocks_premature_booking(self):
        from agent.llm_client import LLMResponse, ToolCall

        # By default enable_dag is True, with prerequisites ['search_flights', 'calculator']
        agent = Agent(model="gpt-4o-mini", enable_dag=True)

        class MockClientPrematureBook:
            def __init__(self):
                self.calls = 0

            def chat(self, messages, tools=None, temperature=0.2):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_premature",
                                name="book_flight",
                                arguments={"flight_number": "BA 178"},
                                raw_arguments='{"flight_number": "BA 178"}',
                            )
                        ],
                    )
                return LLMResponse(content="I cannot book yet.")

        agent.client = MockClientPrematureBook()
        agent.run("Book flight immediately", max_steps=2)
        tool_obs = [m for m in agent.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_obs), 1)
        self.assertIn("DAG Guardrail Rejection", tool_obs[0]["content"])
        self.assertIn("calculator", tool_obs[0]["content"])
        self.assertIn("search_flights", tool_obs[0]["content"])


if __name__ == "__main__":
    unittest.main()

