#!/usr/bin/env python3
"""
Example 2: Complete Travel & Packing Advisor Agent
Demonstrates multi-step reasoning where the agent chains together:
1. Flight search (NYC -> LON)
2. Baggage policy lookup
3. Math calculation for 2 passengers + baggage
4. Destination weather forecast for London
5. Synthesizing everything into a trip itinerary and packing list
"""

import sys
from agent import Agent

def main():
    print("\n" + "=" * 60)
    print(" 🌍 EXAMPLE 2: COMPLETE TRIP & PACKING ADVISOR AGENT")
    print("=" * 60)

    agent = Agent(model="gpt-4o-mini")

    prompt = (
        "We are planning a trip from New York (NYC) to London on 2026-09-25 for 2 travelers with 2 checked bags. "
        "Please: "
        "1. Find available flights and recommend the best value. "
        "2. Check the airline's baggage fees and calculate our grand total cost. "
        "3. Check the weather in London on our arrival date and advise us what clothing to pack."
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
