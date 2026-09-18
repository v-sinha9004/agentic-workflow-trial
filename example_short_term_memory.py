#!/usr/bin/env python3
"""
Example 5: Short-Term Working Memory in Agentic Workflows
Demonstrates:
1. Dialogue Buffer: Tracking conversation turns.
2. Working Scratchpad: Explicit key-value notes for critical facts.
3. Sliding Window Pruning: Keeping recent context within limits without breaking tool-call atomicity.
4. JSON Persistence: Live-saving memory to 'session_memory.json' for easy inspection.
"""

import json
import os
import sys
from agent import Agent, ShortTermMemory


def separator(title: str):
    print("\n" + "=" * 65)
    print(f" 🧠 {title}")
    print("=" * 65)


def main():
    separator("EXAMPLE 5: SHORT-TERM MEMORY & WORKING SCRATCHPAD")
    print("""
Short-term memory solves two fundamental challenges in agent engineering:
1. Context Limits: An LLM cannot process infinite history. A sliding window buffer
   retains the most recent K messages.
2. Fact Retention: Raw chat messages eventually roll off the sliding window. A dedicated
   working scratchpad pins key facts (e.g. user name, destination, preferred airline)
   so the agent never forgets them.
""")

    # -------------------------------------------------------------
    # 1. Understanding Working Memory & Scratchpad Notes
    # -------------------------------------------------------------
    separator("Step 1: Scratchpad Working Notes (Key Facts)")
    memory = ShortTermMemory(
        max_messages=6,
        storage_path="session_memory.json",
        auto_persist=True,
    )
    # Start clean for this demonstration
    memory.clear()

    print("Storing verified user travel constraints into working memory...")
    memory.set_note("user_name", "Vishal")
    memory.set_note("preferred_airline", "British Airways")
    memory.set_note("destination", "London")
    memory.set_note("max_budget", "$600")

    print("\nRendered Short-Term Memory:")
    print(memory.render_ascii())

    # -------------------------------------------------------------
    # 2. Inspecting the JSON File on Disk
    # -------------------------------------------------------------
    separator("Step 2: Live Disk Persistence ('session_memory.json')")
    print("Because auto_persist=True, memory was saved to 'session_memory.json' on disk.")
    if os.path.exists("session_memory.json"):
        with open("session_memory.json", "r") as f:
            content = json.load(f)
        print("\nSnippet of 'session_memory.json' on disk:")
        print(json.dumps({
            "working_notes": content.get("working_notes"),
            "message_count": content.get("message_count"),
        }, indent=2))

    # -------------------------------------------------------------
    # 3. Sliding Window Pruning in Action
    # -------------------------------------------------------------
    separator("Step 3: Sliding Window Buffer & Pruning")
    print("Configured capacity: max_messages = 6")
    print("Simulating 4 conversation turns (8 messages: user + assistant)...\n")

    conversations = [
        ("user", "Hello! Can you help me plan a trip?"),
        ("assistant", "Hello Vishal! Yes, I see you are planning a trip to London."),
        ("user", "What are the baggage policies for British Airways?"),
        ("assistant", "British Airways allows 1 carry-on free and $60 per checked bag."),
        ("user", "Are there flights under $600?"),
        ("assistant", "Yes, BA 178 is available for $450."),
        ("user", "What is the weather like in London?"),
        ("assistant", "London weather is currently Mild & Rainy, around 16°C."),
    ]

    for role, text in conversations:
        if role == "user":
            memory.add_user_message(text)
        else:
            memory.add_assistant_message(text)

    print("Dialogue buffer after 8 messages (pruned to stay within limit of 6):")
    print(memory.render_ascii())

    print("\n💡 NOTICE:")
    print(" • Oldest messages ('Hello! Can you help me plan a trip?') rolled off the sliding window.")
    print(" • BUT all 4 working scratchpad notes (user_name, preferred_airline, etc.) REMAIN INTACT!")
    print(" • The agent retains vital facts without bloating the context window.\n")

    # -------------------------------------------------------------
    # 4. System Prompt Context Injection
    # -------------------------------------------------------------
    separator("Step 4: Dynamic System Context Injection")
    base_prompt = "You are a Flight Assistant."
    injected_prompt = memory.format_system_context(base_prompt)
    print("How the LLM sees the system prompt with short-term memory injected:")
    print("-" * 65)
    print(injected_prompt)
    print("-" * 65)

    # -------------------------------------------------------------
    # 5. Live Multi-Turn Interactive Agent Demonstration
    # -------------------------------------------------------------
    separator("Step 5: Live Agent Multi-Turn Memory Test")
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_api_key:
        print("ℹ️  No OPENAI_API_KEY detected in environment.")
        print("To run the live ReAct test, set your key in .env or run:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python3 example_short_term_memory.py\n")
        print("Everything demonstrated above proves short-term memory, sliding window,")
        print("and JSON persistence are fully operational!")
        return

    print("Running a 2-turn live conversation with Agent...")
    agent = Agent(model="gpt-4o-mini", max_memory_messages=10)
    agent.reset()

    # Turn 1: User introduces facts
    print("\n--- Turn 1: Introducing user and preferences ---")
    turn_1_prompt = "My name is Vishal. I am looking for a flight to London and I prefer British Airways."
    ans_1 = agent.run(turn_1_prompt)

    # Turn 2: User asks agent to recall information
    print("\n--- Turn 2: Asking agent to recall without repeating preferences ---")
    turn_2_prompt = "What is my name and which airline do I prefer?"
    ans_2 = agent.run(turn_2_prompt)

    print("\nAgent Memory State after 2 Turns:")
    print(agent.memory.render_ascii())

    print("\n" + "=" * 65)
    print(" Short-Term Memory demonstration complete!")
    print(" Inspect 'session_memory.json' in your workspace to see the saved snapshot.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
