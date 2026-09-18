#!/usr/bin/env python3
"""
Interactive CLI for the Agentic Workflow.
Chat with your agent in real-time and watch it plan and execute tools.
"""

import sys
from agent import Agent, registry

def print_help():
    print("\nCommands:")
    print("  tools      - List all registered tools and descriptions")
    print("  reset      - Clear agent memory")
    print("  quit/exit  - Exit interactive mode")
    print("  help       - Show this message\n")

def main():
    print("\n" + "=" * 60)
    print(" 🤖 INTERACTIVE AGENTIC WORKFLOW CLI")
    print("=" * 60)
    print("Type your request or 'help' for commands. Type 'quit' to exit.\n")

    agent = Agent()

    while True:
        try:
            user_input = input("\n\033[1mYou > \033[0m").strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! ✈️")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "tools":
                print("\nRegistered Tools:")
                for t in registry.list_tools():
                    print(f" • \033[1m{t.name}\033[0m: {t.description}")
                continue

            if user_input.lower() == "reset":
                agent.reset()
                print("Agent memory reset.")
                continue

            agent.run(user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except PermissionError as e:
            print(f"\n❌ {e}")
            print("Tip: Add your OpenAI API key to a .env file or run: export OPENAI_API_KEY='your-key'\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
