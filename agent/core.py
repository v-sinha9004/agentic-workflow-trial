"""
Agent Controller.
Implements the Plan-and-Execute (ReAct) loop using OpenAI Function Calling.
Maintains working memory, executes tools, logs steps, and synthesizes answers.
"""

from typing import Any, Dict, List, Optional
from .config import Config, load_config
from .llm_client import OpenAIClient, ToolCall
from .tools import ToolRegistry, registry as default_registry


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

Your goal is to help users find flights, check baggage policies, calculate total trip costs, and provide destination advice.

Guidelines for Planning and Tool Use:
1. Always plan your actions step-by-step.
2. When a user asks about flights, search for options using 'search_flights'.
3. Always verify airline baggage rules and fees using 'get_baggage_policy' when passengers or luggage are involved.
4. Always use 'calculator' to compute accurate total costs (tickets + baggage + passengers). Never guess math.
5. If the user mentions travel dates or asks for recommendations, check destination weather with 'get_city_weather'.
6. Present your final answer in a clear, well-structured, and helpful format.
"""


class Agent:
    """
    ReAct Agent orchestrator.
    Manages the Thought -> Action -> Observation cycle until task completion.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        config: Optional[Config] = None,
        verbose: bool = True,
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
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self):
        """Clears working memory while keeping the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_prompt: str, max_steps: int = 10) -> str:
        """
        Runs the agent on the given user prompt.
        Executes iterative ReAct loop:
          1. Send message history & tool schemas to OpenAI
          2. If LLM requests tool execution: execute tools and append observations
          3. Repeat until LLM generates final answer or max_steps reached.
        """
        # Append user prompt to conversation history
        self.messages.append({"role": "user", "content": user_prompt})

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
                # Format and record the assistant's message containing tool_calls in history
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                self.messages.append(assistant_msg)

                if self.verbose and response.content:
                    print(f"{Colors.DIM}Thought: {response.content}{Colors.RESET}")

                # Execute each tool requested by the model
                for tc in response.tool_calls:
                    if self.verbose:
                        print(f"{Colors.CYAN}⚙️  Action (Tool Call):{Colors.RESET} {Colors.BOLD}{tc.name}{Colors.RESET}({tc.arguments})")

                    tool_obj = self.registry.get(tc.name)
                    if not tool_obj:
                        observation = f"Error: Tool '{tc.name}' is not registered."
                    else:
                        observation = tool_obj.execute(**tc.arguments)

                    if self.verbose:
                        # Print truncated preview if observation is very long
                        obs_preview = observation if len(observation) < 350 else observation[:350] + "..."
                        print(f"{Colors.YELLOW}👁️  Observation:{Colors.RESET} {obs_preview}\n")

                    # Append observation as a tool role message in memory
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    })

                # Loop back to let the LLM observe results and decide the next step
                continue

            # Case 2: The model produced a final answer without requesting more tools
            final_answer = response.content or ""
            self.messages.append({"role": "assistant", "content": final_answer})

            if self.verbose:
                print(f"{Colors.BOLD}{Colors.GREEN}======================================================={Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.GREEN}🎯 FINAL ANSWER ({step} steps):{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.GREEN}======================================================={Colors.RESET}")
                print(f"\n{final_answer}\n")

            return final_answer

        fallback_msg = f"Agent reached maximum allowed steps ({max_steps}) before concluding."
        if self.verbose:
            print(f"{Colors.RED}{fallback_msg}{Colors.RESET}")
        return fallback_msg
