# Educational Agentic Workflow: Plan-and-Execute (ReAct) with Tools

A lightweight, transparent, from-scratch Python framework demonstrating how modern AI agents think, plan, and call tools to accomplish real-world goals.

Powered by **OpenAI Function Calling** and the **ReAct (Reason + Act)** pattern.

---

## 🧠 Core Concept: What is an Agentic Workflow?

A standard LLM call is **one-shot**: you give it a prompt, and it gives you text back. It cannot browse current databases, perform exact math calculations reliably, or take external actions.

An **Agentic Workflow** gives the LLM tools and a feedback loop:

```mermaid
flowchart TD
    User["User Goal"] --> LLM["LLM (Planning & Reasoning)"]
    LLM -- "Needs more information" --> ToolCall["Tool Call: name(arguments)"]
    ToolCall --> Executor["Execute Python Function"]
    Executor --> Observation["Observation: Tool Output"]
    Observation --> LLM
    LLM -- "Has sufficient info" --> FinalAnswer["Final Answer to User"]
```

### The ReAct (Reason + Act) Loop:
1. **Reason (Thought)**: The model determines what information is missing to solve the user's goal.
2. **Act (Action)**: The model selects a registered tool (e.g. `search_flights`) and provides the required parameters.
3. **Observe (Observation)**: The environment executes the Python function and feeds the real data back to the LLM.
4. **Iterate**: The LLM inspects the observation and either calls another tool (e.g. `get_baggage_policy` or `calculator`) or synthesizes the final answer.

---

## 🗂️ Project Structure

```
agentic-workflow-trial/
├── agent/
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration loader (.env, OPENAI_API_KEY, model)
│   ├── tools.py            # ToolRegistry, @tool decorator, flight/travel tools
│   ├── llm_client.py       # Direct OpenAI API client with function calling
│   └── core.py             # Agent class running the ReAct loop
├── examples/
│   ├── example_flight_search.py   # Flight search + baggage fare calculation
│   ├── example_trip_planner.py    # Multi-step: flights + baggage + weather advice
│   └── example_human_in_loop.py   # Human-in-the-Loop verification & booking
├── tests/
│   ├── test_tools.py       # Unit tests for schemas and tool execution
│   └── test_agent_dry_run.py # Unit tests for configuration and agent setup
├── interactive_cli.py      # Real-time interactive terminal chat with agent
├── .env.example            # Environment template
└── README.md               # Educational guide
```

---

## 🚀 Getting Started

### 1. Set Up with `uv`
This project is configured for **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management:
```bash
# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install openai python-dotenv

# Or install from pyproject.toml / requirements.txt:
uv pip install -r requirements.txt
```

### 2. Configure your OpenAI API Key
Create a `.env` file from the template:
```bash
cp .env.example .env
```
Open `.env` and set your key:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```
*(You can also export in your terminal: `export OPENAI_API_KEY="sk-..."`).*

### 3. Run the Examples

#### Example 1: Flight Search & Fare Calculation
```bash
uv run example_flight_search.py
# or with active venv:
python3 example_flight_search.py
```

#### Example 2: Complete Trip & Packing Advisor (Multi-Step)
```bash
uv run example_trip_planner.py
# or with active venv:
python3 example_trip_planner.py
```

#### Example 3: Human-in-the-Loop Flight Verification & Booking
```bash
uv run example_human_in_loop.py
# or with active venv:
python3 example_human_in_loop.py
```

#### Interactive Terminal Chat
Chat with your travel agent live and watch it plan and call tools in real-time:
```bash
uv run interactive_cli.py
# or with active venv:
python3 interactive_cli.py
```

---

## 🛠️ Built-in Tools

| Tool | Description | Parameters |
|---|---|---|
| `search_flights` | Search available flight routes, airlines, and base ticket prices | `origin`, `destination`, `date` |
| `get_baggage_policy` | Look up carry-on allowance and checked bag fees per carrier | `airline` |
| `calculator` | Safely evaluate mathematical expressions for fares, fees, and taxes | `expression` |
| `get_city_weather` | Check current conditions and travel packing recommendations | `city`, `date` |
| `book_flight` | Book confirmed flight tickets once human verification is provided | `flight_number`, `passenger_name`, `date` |

---

## 💡 How to Add Your Own Custom Tools

Adding a new tool is as simple as writing a regular Python function with type annotations and a docstring, then decorating it with `@tool`:

```python
from agent import tool

@tool(name="find_hotel", description="Search hotels in a city within a price budget.")
def find_hotel(city: str, max_price_per_night: float) -> str:
    """
    Search available hotels.
    :param city: Destination city name
    :param max_price_per_night: Maximum budget per night in USD
    """
    # Your database lookup or API call here
    return f"Found Grand Hotel in {city} for ${max_price_per_night - 20}/night."
```

The `@tool` decorator **automatically**:
1. Inspects the parameter names and Python type hints (`str`, `int`, `float`, `bool`).
2. Parses parameter descriptions from the docstring.
3. Generates the exact **OpenAI JSON Function Schema** that OpenAI's API requires.

---

## 🧪 Running Unit Tests

Run the test suite to verify tool schema generation, safe calculation, and agent configuration:
```bash
python3 -m unittest discover -s tests
```

---

## 🔭 Next Steps & Expansion Ideas

When you are ready to expand this framework, here are great concepts to explore:
1. **Persistent Memory**: Save conversation history to SQLite or JSON so the agent remembers past trips across sessions.
2. **Human-in-the-Loop**: Pause the agent before executing sensitive tools (like `book_ticket(credit_card=...)`) to request user confirmation.
3. **Live API Integration**: Replace mock flight/weather data with real APIs (e.g. Amadeus/Skyscanner, OpenWeatherMap).
4. **Multi-Agent Teams**: Have a "Researcher Agent" gather flights and a "Budget Agent" audit costs.
