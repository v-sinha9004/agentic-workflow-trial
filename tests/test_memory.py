import os
import unittest
import json
from agent.memory import ShortTermMemory
from agent.core import Agent
from agent.llm_client import LLMResponse, ToolCall
from agent.tools import save_memory_note, recall_memory_notes, set_active_memory


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.test_storage = "test_temp_memory.json"
        if os.path.exists(self.test_storage):
            os.remove(self.test_storage)

    def tearDown(self):
        if os.path.exists(self.test_storage):
            os.remove(self.test_storage)

    def test_basic_message_tracking(self):
        mem = ShortTermMemory(max_messages=10, storage_path=None)
        mem.add_user_message("Hello world")
        mem.add_assistant_message("Hi there! How can I assist you?")
        mem.add_tool_message(tool_call_id="tc_1", content="Tool result", name="search_flights")

        self.assertEqual(len(mem.messages), 3)
        self.assertEqual(mem.messages[0]["role"], "user")
        self.assertEqual(mem.messages[0]["content"], "Hello world")
        self.assertEqual(mem.messages[1]["role"], "assistant")
        self.assertEqual(mem.messages[2]["role"], "tool")
        self.assertEqual(mem.messages[2]["tool_call_id"], "tc_1")

    def test_working_scratchpad_notes(self):
        mem = ShortTermMemory(max_messages=10, storage_path=None)
        mem.set_note("user_name", "Vishal")
        mem.set_note("airline", "British Airways")

        self.assertEqual(mem.get_note("user_name"), "Vishal")
        self.assertEqual(mem.get_note("airline"), "British Airways")
        self.assertIsNone(mem.get_note("unknown_key"))
        self.assertEqual(mem.get_note("unknown_key", "default_val"), "default_val")

        all_notes = mem.get_all_notes()
        self.assertEqual(len(all_notes), 2)

        mem.delete_note("airline")
        self.assertIsNone(mem.get_note("airline"))
        self.assertEqual(len(mem.get_all_notes()), 1)

    def test_sliding_window_pruning(self):
        mem = ShortTermMemory(max_messages=4, storage_path=None)
        for i in range(1, 7):
            mem.add_user_message(f"Message {i}")

        self.assertEqual(len(mem.messages), 4)
        # Should keep Message 3, 4, 5, 6
        self.assertEqual(mem.messages[0]["content"], "Message 3")
        self.assertEqual(mem.messages[-1]["content"], "Message 6")

    def test_tool_call_atomicity_during_pruning(self):
        mem = ShortTermMemory(max_messages=3, storage_path=None)
        mem.add_user_message("msg 1")
        # Assistant makes a tool call
        mem.add_assistant_message(
            content="Thinking...",
            tool_calls=[{"id": "call_abc", "function": {"name": "test_tool"}}],
        )
        # Tool response arrives
        mem.add_tool_message(tool_call_id="call_abc", content="Tool output 1")
        mem.add_user_message("msg 2")

        # Total added = 4. With max_messages=3, excess is 1.
        # Simple cut at index 1 would leave index 1 (assistant), which is fine.
        # But if excess would start at a 'tool' message, it must not orphan the tool message.
        self.assertLessEqual(len(mem.messages), 3)
        # Verify no orphaned tool message at index 0 without preceding assistant
        if mem.messages and mem.messages[0]["role"] == "tool":
            self.fail("Pruning left an orphaned tool message at the start of dialogue history.")

    def test_system_prompt_context_injection(self):
        mem = ShortTermMemory(max_messages=10, storage_path=None)
        base = "You are a helpful assistant."
        self.assertEqual(mem.format_system_context(base), base)

        mem.set_note("user_name", "Vishal")
        mem.set_note("budget", "$500")
        injected = mem.format_system_context(base)

        self.assertIn(base, injected)
        self.assertIn("ACTIVE WORKING MEMORY / SCRATCHPAD", injected)
        self.assertIn("user_name: Vishal", injected)
        self.assertIn("budget: $500", injected)

    def test_json_persistence_save_and_load(self):
        mem1 = ShortTermMemory(max_messages=10, storage_path=self.test_storage, auto_persist=True)
        mem1.set_note("origin", "NYC")
        mem1.add_user_message("Find flights to London")
        mem1.add_assistant_message("Searching flights...")

        self.assertTrue(os.path.exists(self.test_storage))

        with open(self.test_storage, "r") as f:
            data = json.load(f)
        self.assertEqual(data["working_notes"]["origin"], "NYC")
        self.assertEqual(len(data["dialogue_history"]), 2)

        # Load into new memory instance with load_existing=True
        mem2 = ShortTermMemory(storage_path=self.test_storage, load_existing=True)
        self.assertEqual(mem2.get_note("origin"), "NYC")
        self.assertEqual(len(mem2.messages), 2)
        self.assertEqual(mem2.messages[0]["content"], "Find flights to London")

    def test_clear_resets_memory_and_file(self):
        mem = ShortTermMemory(max_messages=10, storage_path=self.test_storage, auto_persist=True)
        mem.set_note("test_key", "test_val")
        mem.add_user_message("test message")
        self.assertTrue(os.path.exists(self.test_storage))

        mem.clear()
        self.assertEqual(len(mem.messages), 0)
        self.assertEqual(len(mem.working_notes), 0)
        self.assertFalse(os.path.exists(self.test_storage))

    def test_render_ascii(self):
        mem = ShortTermMemory(max_messages=10, storage_path=self.test_storage)
        mem.set_note("airline", "Delta")
        mem.add_user_message("Show me Delta flights")
        rendered = mem.render_ascii()

        self.assertIn("SHORT-TERM WORKING MEMORY STATUS", rendered)
        self.assertIn("airline: Delta", rendered)
        self.assertIn("Show me Delta flights", rendered)

    def test_agent_memory_integration(self):
        agent = Agent(
            model="gpt-4o-mini",
            enable_dag=False,
            memory_storage_path=self.test_storage,
        )
        self.assertIsNotNone(agent.memory)
        self.assertEqual(len(agent.messages), 1)  # Only system prompt initially

        # Verify memory tools can be called
        set_active_memory(agent.memory)
        res_save = save_memory_note("seat_type", "Window")
        self.assertIn("Stored in working memory", res_save)
        self.assertEqual(agent.memory.get_note("seat_type"), "Window")

        res_recall = recall_memory_notes()
        self.assertIn("seat_type", res_recall)
        self.assertIn("Window", res_recall)

        # Resetting agent clears memory
        agent.reset()
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(len(agent.memory.working_notes), 0)


if __name__ == "__main__":
    unittest.main()
