"""
Agent Controller.
Implements the Plan-and-Execute (ReAct) loop using OpenAI Function Calling.
Maintains working memory, executes tools, logs steps, and synthesizes answers.
"""

from typing import Any, Callable, Dict, List, Optional
from .config import Config, load_config
from .llm_client import OpenAIClient, ToolCall
from .tools import ToolRegistry, registry as default_registry, set_active_memory
from .dag import WorkflowDAG, create_flight_booking_dag
from .memory import ShortTermMemory


# Terminal color formatting for clear educational visualization
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


DEFAULT_SYSTEM_PROMPT = """You are an intelligent, proactive Travel & Flight Assistant powered by an agentic workflow.

Your goal is to help users find flights, check baggage policies, calculate total trip costs, book confirmed flights, and provide destination advice.

Guidelines for Planning and Tool Use:
1. Always plan your actions step-by-step following the Workflow DAG dependencies:
   - Step 1: 'search_flights' -> Search for flights and airline options.
   - Step 2: 'get_baggage_policy' & 'calculator' -> Verify baggage rules and calculate total cost.
   - Step 3: Human Verification -> Present flight details, baggage fees, and total cost to the user.
   - Step 4: 'book_flight' -> Book only AFTER flight search and cost calculation are completed, AND user explicitly confirms.
2. When a user asks about flights, search for options using 'search_flights'.
3. Always verify airline baggage rules and fees using 'get_baggage_policy' when passengers or luggage are involved.
4. Always use 'calculator' to compute accurate total costs (tickets + baggage + passengers). Never guess math.
5. If the user mentions travel dates or asks for recommendations, check destination weather with 'get_city_weather'.
6. SHORT-TERM MEMORY & WORKING NOTES:
   - When the user shares important personal details, travel preferences, or constraints (e.g. passenger name, destination, preferred airline, budget, baggage count), save them to working memory using 'save_memory_note'.
   - Use stored working memory facts to personalize responses and avoid repeatedly asking the user for information they already provided.
7. HUMAN-IN-THE-LOOP VERIFICATION FOR FLIGHT BOOKINGS:
   - When presenting flight options, ALWAYS show the flight details, airline, schedule, and complete cost breakdown first.
   - NEVER call 'book_flight' without explicit user confirmation.
   - After displaying the flight details, explicitly ask the user for confirmation (e.g. "Would you like me to book this flight? Please reply 'yes book' to confirm.").
   - Only call 'book_flight' once the user explicitly confirms (e.g. "yes book", "yes, book it", or explicitly instructs to book).
8. Present your final answer in a clear, well-structured, and helpful format.
"""


class Agent:
    """
    ReAct Agent orchestrator.
    Manages the Thought -> Action -> Observation cycle until task completion.
    Enforces step dependencies via a Directed Acyclic Graph (DAG).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        config: Optional[Config] = None,
        verbose: bool = True,
        confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        dag: Optional[WorkflowDAG] = None,
        enable_dag: bool = True,
        memory: Optional[ShortTermMemory] = None,
        max_memory_messages: int = 20,
        memory_storage_path: Optional[str] = "session_memory.json",
        load_existing_memory: bool = False,
    ):
        self.config = config or load_config(model_override=model)
        self.client = OpenAIClient(
            api_key=self.config.api_key,
            model=self.config.model,
            base_url=self.config.base_url,
        )
        self.registry = tool_registry or default_registry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.confirmation_callback = confirmation_callback
        self.enable_dag = enable_dag
        self.dag = (dag if dag is not None else create_flight_booking_dag()) if enable_dag else None
        self.memory = memory or ShortTermMemory(
            max_messages=max_memory_messages,
            storage_path=memory_storage_path,
            load_existing=load_existing_memory,
        )
        set_active_memory(self.memory)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Active dialogue context with dynamically formatted system prompt including working notes."""
        system_msg = {
            "role": "system",
            "content": self.memory.format_system_context(self.system_prompt),
        }
        return [system_msg] + self.memory.messages

    @messages.setter
    def messages(self, msgs: List[Dict[str, Any]]):
        """Allows direct message assignment for test mocking and backward compatibility."""
        self.memory.messages = [m for m in msgs if m.get("role") != "system"]

    def reset(self):
        """Clears working memory while keeping the system prompt, and resets the DAG."""
        self.memory.clear()
        if self.dag:
            self.dag.reset()

    def run(self, user_prompt: str, max_steps: int = 10) -> str:
        """
        Runs the agent on the given user prompt.
        Executes iterative ReAct loop:
          1. Send message history & tool schemas to OpenAI
          2. If LLM requests tool execution: execute tools and append observations
          3. Repeat until LLM generates final answer or max_steps reached.
        """
        set_active_memory(self.memory)

        # Append user prompt to working memory
        self.memory.add_user_message(user_prompt)

        if self.verbose:
            print(f"\n{Colors.BOLD}{Colors.HEADER}======================================================={Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.HEADER} AGENTIC WORKFLOW: PLAN & EXECUTE LOOP{Colors.RESET}")
            print(f"{Colors.DIM} Model: {self.config.model} | Max Steps: {max_steps}{Colors.RESET}")
            print(f"{Colors.BOLD} User Goal:{Colors.RESET} {user_prompt}")
            print(f"{Colors.BOLD}{Colors.HEADER}======================================================={Colors.RESET}\n")

        for step in range(1, max_steps + 1):
            if self.verbose:
                print(f"{Colors.BOLD}{Colors.BLUE}--- [Step {step}] LLM Planning & Reasoning ---{Colors.RESET}")

            tool_schemas = self.registry.get_schemas()

            # Call OpenAI with current conversation history and available tools
            response = self.client.chat(messages=self.messages, tools=tool_schemas)

            # Case 1: The model decided to invoke one or more tools
            if response.tool_calls:
                # Format and record the assistant's message containing tool_calls in memory
                assistant_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]
                self.memory.add_assistant_message(
                    content=response.content,
                    tool_calls=assistant_tool_calls,
                )

                if self.verbose and response.content:
                    print(f"{Colors.DIM}Thought: {response.content}{Colors.RESET}")

                # Execute each tool requested by the model
                for tc in response.tool_calls:
                    if self.verbose:
                        print(f"{Colors.CYAN}⚙️  Action (Tool Call):{Colors.RESET} {Colors.BOLD}{tc.name}{Colors.RESET}({tc.arguments})")

                    tool_obj = self.registry.get(tc.name)
                    if not tool_obj:
                        observation = f"Error: Tool '{tc.name}' is not registered."
                    elif self.dag and not self.dag.can_execute(tc.name)[0]:
                        # DAG step dependency check failed
                        _, unmet_deps = self.dag.can_execute(tc.name)
                        observation = (
                            f"DAG Guardrail Rejection: Tool '{tc.name}' cannot be executed yet because "
                            f"its prerequisite steps are not completed. Unmet dependencies: {unmet_deps}. "
                            f"Please execute the prerequisite steps first."
                        )
                        if self.verbose:
                            print(f"{Colors.RED}🚫 DAG Blocked:{Colors.RESET} {tc.name} requires {unmet_deps}")
                    elif tool_obj.requires_confirmation and self.confirmation_callback:
                        # Human-in-the-loop verification check
                        if not self.confirmation_callback(tc.name, tc.arguments):
                            observation = (
                                f"Action cancelled: Human verification declined for tool '{tc.name}' "
                                f"with arguments {tc.arguments}."
                            )
                        else:
                            observation = tool_obj.execute(**tc.arguments)
                            if self.dag and not observation.startswith("Error"):
                                self.dag.mark_completed(tc.name, observation)
                    else:
                        observation = tool_obj.execute(**tc.arguments)
                        if self.dag and not observation.startswith("Error"):
                            self.dag.mark_completed(tc.name, observation)

                    if self.verbose:
                        # Print truncated preview if observation is very long
                        obs_preview = observation if len(observation) < 350 else observation[:350] + "..."
                        print(f"{Colors.YELLOW}👁️  Observation:{Colors.RESET} {obs_preview}\n")

                    # Append observation as a tool role message in memory
                    self.memory.add_tool_message(
                        tool_call_id=tc.id,
                        content=observation,
                        name=tc.name,
                    )

                # Loop back to let the LLM observe results and decide the next step
                continue

            # Case 2: The model produced a final answer without requesting more tools
            final_answer = response.content or ""
            self.memory.add_assistant_message(content=final_answer)

            if self.verbose:
                print(f"{Colors.BOLD}{Colors.GREEN}======================================================={Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.GREEN}🎯 FINAL ANSWER ({step} steps):{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.GREEN}======================================================={Colors.RESET}")
                print(f"\n{final_answer}\n")

            return final_answer

        fallback_msg = f"Agent reached maximum allowed steps ({max_steps}) before concluding."
        self.memory.add_assistant_message(content=fallback_msg)
        if self.verbose:
            print(f"{Colors.RED}{fallback_msg}{Colors.RESET}")
        return fallback_msg
