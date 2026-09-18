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


if __name__ == "__main__":
    unittest.main()
