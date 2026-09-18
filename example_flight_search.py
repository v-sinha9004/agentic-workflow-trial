#!/usr/bin/env python3
"""
Example 1: Flight Search & Fare Calculation Agent
Demonstrates how the agent:
1. Searches for flights between two cities.
2. Identifies the cheapest carrier.
3. Looks up baggage policies.
4. Uses the calculator to accurately calculate total cost for 2 passengers + luggage.
"""

import sys
import os
from agent import Agent

def main():
    print("\n" + "=" * 60)
    print(" 🛫 EXAMPLE 1: FLIGHT SEARCH & FARE CALCULATION AGENT")
    print("=" * 60)

    # You can specify any OpenAI model: "gpt-4o-mini", "gpt-4o", etc.
    # It will also read OPENAI_MODEL and OPENAI_API_KEY from .env
    agent = Agent(model="gpt-4o-mini")

    prompt = (
        "I need a flight from NYC to London for 2 passengers on 2026-09-25. "
        "Find the cheapest option, check baggage fees for that airline, and "
        "calculate the exact total cost if we take 2 checked bags in total."
    )

    try:
        agent.run(prompt)
    except PermissionError as e:
        print(f"\n❌ {e}")
        print("Tip: Add your OpenAI API key to a .env file or run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
