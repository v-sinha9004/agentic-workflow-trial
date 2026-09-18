#!/usr/bin/env python3
"""
Example 3: Human-in-the-Loop (HITL) Flight Verification & Booking
Demonstrates how the agentic workflow incorporates human verification:
1. Turn 1: User asks to search and recommend a flight.
   - Agent calls search_flights, get_baggage_policy, and calculator.
   - Agent presents the flight details and pricing, then PAUSES to ask for human confirmation.
   - Agent does NOT book yet.
2. Turn 2: User explicitly confirms with "yes book".
   - Agent observes the user's approval.
   - Agent calls book_flight to finalize the reservation.
   - Agent provides the confirmed booking reference and ticket details.
"""

import sys
from agent import Agent


def main():
    print("\n" + "=" * 65)
    print(" 👤 EXAMPLE 3: HUMAN-IN-THE-LOOP (HITL) FLIGHT VERIFICATION")
    print("=" * 65)

    agent = Agent(model="gpt-4o-mini")

    # Turn 1: User requests flight details
    print("\n" + "-" * 65)
    print(" [Turn 1] User Request: Find flights and provide details")
    print("-" * 65)
    turn1_prompt = (
        "Find the best flight from NYC to London on 2026-09-25 for passenger Alice Smith. "
        "Show me the flight details, baggage policy, and total cost."
    )

    try:
        agent.run(turn1_prompt)

        # Turn 2: Human in the loop provides confirmation
        print("\n" + "-" * 65)
        print(" [Turn 2] Human Verification: User confirms booking ('yes book')")
        print("-" * 65)
        turn2_prompt = "yes book"
        agent.run(turn2_prompt)

    except PermissionError as e:
        print(f"\n❌ {e}")
        print("Tip: Add your OpenAI API key to a .env file or run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
