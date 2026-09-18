"""
Short-Term Working Memory Engine for Agentic Workflows.

Provides:
1. Dialogue History Buffer: Tracks conversational turns with sliding window pruning.
2. Tool-Call Atomicity: Ensures assistant tool_calls and tool observations are never split apart during pruning.
3. Explicit Working Scratchpad: Key-value store for facts decided by the agent or user.
4. JSON Persistence: Saves memory to disk (e.g., session_memory.json) in real time for inspection.
5. Educational ASCII Visualization: Inspect memory capacity, turns, and notes at any time.
"""

import json
import os
from typing import Any, Dict, List, Optional


class ShortTermMemory:
    """
    Manages short-term working memory for an AI Agent.
    """

    def __init__(
        self,
        max_messages: int = 20,
        storage_path: Optional[str] = "session_memory.json",
        auto_persist: bool = True,
        load_existing: bool = False,
    ):
        """
        Args:
            max_messages: Maximum number of conversation messages to keep (excluding system prompt).
            storage_path: Path to the JSON file where memory is saved (None to disable disk storage).
            auto_persist: Whether to automatically save to storage_path on every change.
            load_existing: Whether to load previously saved memory from storage_path on initialization.
        """
        self.max_messages = max_messages
        self.storage_path = storage_path
        self.auto_persist = auto_persist

        self.messages: List[Dict[str, Any]] = []
        self.working_notes: Dict[str, Any] = {}

        # Only load existing session if explicitly requested
        if load_existing and self.storage_path and os.path.exists(self.storage_path):
            try:
                self.load_from_file(self.storage_path)
            except Exception:
                # If reading fails or file is empty, start clean
                self.messages = []
                self.working_notes = {}

    def add_user_message(self, content: str):
        """Records a message from the user."""
        self.messages.append({"role": "user", "content": content})
        self._after_update()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        """Records a response or tool call decision from the assistant."""
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self._after_update()

    def add_tool_message(self, tool_call_id: str, content: str, name: Optional[str] = None):
        """Records an observation/result from a tool execution."""
        msg: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        if name:
            msg["name"] = name
        self.messages.append(msg)
        self._after_update()

    def set_note(self, key: str, value: Any):
        """Stores or updates a key fact in the working scratchpad."""
        self.working_notes[key] = value
        self._after_update()

    def get_note(self, key: str, default: Any = None) -> Any:
        """Retrieves a key fact from the working scratchpad."""
        return self.working_notes.get(key, default)

    def get_all_notes(self) -> Dict[str, Any]:
        """Returns all working scratchpad notes."""
        return dict(self.working_notes)

    def delete_note(self, key: str):
        """Removes a key fact from the working scratchpad."""
        if key in self.working_notes:
            del self.working_notes[key]
            self._after_update()

    def clear(self):
        """Clears all conversation messages and working notes."""
        self.messages = []
        self.working_notes = {}
        self._after_update()
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except OSError:
                pass

    def prune(self):
        """
        Enforces the sliding window limit (max_messages) while strictly maintaining
        OpenAI tool-call atomicity.

        Rule: An assistant message containing tool_calls and its corresponding
        subsequent 'tool' observation messages must NEVER be separated.
        """
        if len(self.messages) <= self.max_messages:
            return

        excess = len(self.messages) - self.max_messages

        # Find a safe cut point where we don't sever tool_calls from tool responses
        cut_idx = excess
        while cut_idx < len(self.messages):
            # If the candidate at cut_idx is a 'tool' message, we cannot start the active list here,
            # because its parent assistant message would have been pruned.
            if self.messages[cut_idx].get("role") == "tool":
                cut_idx += 1
            else:
                break

        self.messages = self.messages[cut_idx:]

    def _after_update(self):
        """Applies sliding window pruning and persists to JSON file if configured."""
        self.prune()
        if self.auto_persist and self.storage_path:
            self.save_to_file(self.storage_path)

    def save_to_file(self, file_path: str):
        """Saves current short-term memory state to a JSON file."""
        data = {
            "_info": "Short-Term Working Memory Snapshot",
            "max_messages": self.max_messages,
            "message_count": len(self.messages),
            "working_notes": self.working_notes,
            "dialogue_history": self.messages,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str):
        """Loads memory state from a JSON file."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.working_notes = data.get("working_notes", {})
        self.messages = data.get("dialogue_history", [])
        if "max_messages" in data:
            self.max_messages = data["max_messages"]

    def format_system_context(self, base_system_prompt: str) -> str:
        """
        Injects the active working scratchpad notes into the system prompt
        so the model is always explicitly informed of pinned facts.
        """
        if not self.working_notes:
            return base_system_prompt

        notes_str = "\n".join(f"  • {k}: {v}" for k, v in self.working_notes.items())
        memory_section = (
            f"\n\n--- [ACTIVE WORKING MEMORY / SCRATCHPAD] ---\n"
            f"The following verified facts have been stored in short-term working memory:\n"
            f"{notes_str}\n"
            f"Use these facts when reasoning and answering without repeatedly asking the user."
        )
        return base_system_prompt + memory_section

    def render_ascii(self) -> str:
        """Returns a formatted ASCII diagram of current short-term memory state."""
        bar = "─" * 60
        lines = [
            f"┌{bar}┐",
            f"│  🧠 SHORT-TERM WORKING MEMORY STATUS                      │",
            f"├{bar}┤",
            f"│  Storage: {self.storage_path or 'RAM (ephemeral)' :<48}│",
            f"│  Capacity: {len(self.messages)} / {self.max_messages} messages (Sliding Window)            │",
            f"├{bar}┤",
            f"│  📌 WORKING SCRATCHPAD NOTES ({len(self.working_notes)} active facts):                 │",
        ]

        if not self.working_notes:
            lines.append(f"│    (No active notes stored yet)                           │")
        else:
            for k, v in self.working_notes.items():
                entry = f"• {k}: {v}"
                if len(entry) > 54:
                    entry = entry[:51] + "..."
                lines.append(f"│    {entry:<55}│")

        lines.append(f"├{bar}┤")
        lines.append(f"│  💬 RECENT DIALOGUE BUFFER ({len(self.messages)} messages):                     │")

        if not self.messages:
            lines.append(f"│    (Dialogue buffer empty)                                │")
        else:
            for i, msg in enumerate(self.messages[-8:], 1):  # Show last up to 8 messages
                role = msg.get("role", "unknown").upper()
                content = msg.get("content") or ""
                if "tool_calls" in msg:
                    tc_names = [tc.get("function", {}).get("name", "") for tc in msg["tool_calls"]]
                    content = f"[Tool Call: {', '.join(tc_names)}]"
                snippet = content.replace("\n", " ")
                if len(snippet) > 42:
                    snippet = snippet[:39] + "..."
                lines.append(f"│   [{role[:4]}] {snippet:<50}│")

        lines.append(f"└{bar}┘")
        return "\n".join(lines)
