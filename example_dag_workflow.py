#!/usr/bin/env python3
"""
Example 4: Directed Acyclic Graph (DAG) Step Ordering & Enforcement
Demonstrates how the Workflow DAG guarantees that flight booking cannot happen
before prerequisite steps (flight search, baggage check, cost calculation) are completed.
"""

from agent import Agent, create_flight_booking_dag


def main():
    print("\n" + "=" * 65)
    print(" 📊 EXAMPLE 4: WORKFLOW DAG STEP ORDERING & ENFORCEMENT")
    print("=" * 65)

    # 1. Inspect the initial DAG
    dag = create_flight_booking_dag()
    print("\n1. Initial Workflow DAG Definition:")
    print(dag.render_ascii())

    # 2. Demonstrate DAG Prerequisite Check
    print("\n2. Checking step readiness before search:")
    can_search, _ = dag.can_execute("search_flights")
    can_book, unmet = dag.can_execute("book_flight")
    print(f" • Can execute 'search_flights'? {can_search} (Entry step)")
    print(f" • Can execute 'book_flight'?   {can_book} (Blocked: requires {unmet})")

    # 3. Simulate step progression
    print("\n3. Simulating step completion:")
    print(" -> Executing: search_flights('NYC', 'LON', '2026-09-25')...")
    dag.mark_completed("search_flights")
    print(dag.render_ascii())

    can_book, unmet = dag.can_execute("book_flight")
    print(f"\n • Can execute 'book_flight' now? {can_book} (Still blocked: requires {unmet})")

    print("\n -> Executing: calculator('450 + 60')...")
    dag.mark_completed("calculator")
    print(dag.render_ascii())

    can_book, unmet = dag.can_execute("book_flight")
    print(f"\n • Can execute 'book_flight' now? {can_book} (All dependencies completed! Ready for Human Verification)")

    # 4. Agent Guardrail Demonstration
    print("\n4. Agent DAG Guardrail in Action:")
    agent = Agent(model="gpt-4o-mini", verbose=False)
    print(agent.dag.render_ascii())
    print("\nAttempting direct booking without prior search or calculation...")
    can_run, unmet = agent.dag.can_execute("book_flight")
    if not can_run:
        print(f"🚫 DAG GUARD INTERCEPT: 'book_flight' rejected! Unmet prerequisites: {unmet}")

    print("\n" + "=" * 65)
    print(" DAG demonstration complete. Step ordering is strictly enforced!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
