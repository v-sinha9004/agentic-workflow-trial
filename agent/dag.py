"""
Directed Acyclic Graph (DAG) Workflow Engine.
Provides step dependency management, cycle detection, and prerequisite enforcement
to guarantee that sensitive or dependent tools are never executed out of order.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


class DAGCycleError(Exception):
    """Raised when a circular dependency is detected in the workflow graph."""
    pass


class DAGValidationError(Exception):
    """Raised when an invalid node or dependency is specified."""
    pass


@dataclass
class WorkflowNode:
    name: str
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    completed: bool = False
    result: Optional[str] = None


class WorkflowDAG:
    """
    Manages a Directed Acyclic Graph of workflow steps.
    Enforces that steps cannot execute until all their prerequisite steps have finished.
    """

    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}

    def add_step(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ) -> "WorkflowDAG":
        """
        Registers a step in the DAG along with its prerequisite dependencies.
        Validates that adding this step does not introduce a cycle.
        """
        deps = set(dependencies or [])
        self.nodes[name] = WorkflowNode(name=name, description=description, dependencies=deps)
        self._validate_acyclic()
        return self

    def _validate_acyclic(self):
        """
        Ensures the graph remains a Directed Acyclic Graph (DAG) using Depth First Search (DFS).
        Raises DAGCycleError if a cycle is detected.
        """
        # 0 = unvisited, 1 = currently in recursion stack (visiting), 2 = completely visited
        state: Dict[str, int] = {}

        def dfs(node_name: str, path: List[str]):
            state[node_name] = 1
            node = self.nodes.get(node_name)
            if node:
                for dep in node.dependencies:
                    # Dep might be external or defined later; check if in graph
                    if dep not in self.nodes:
                        continue
                    if state.get(dep) == 1:
                        cycle_path = " -> ".join(path + [dep])
                        raise DAGCycleError(f"Circular dependency detected in DAG: {cycle_path}")
                    if state.get(dep, 0) == 0:
                        dfs(dep, path + [dep])
            state[node_name] = 2

        for name in list(self.nodes.keys()):
            if state.get(name, 0) == 0:
                dfs(name, [name])

    def can_execute(self, name: str) -> Tuple[bool, List[str]]:
        """
        Checks whether a step is eligible to execute.
        Returns (is_ready, list_of_unmet_dependencies).
        If the step is not registered in the DAG, it is considered unconstrained (returns True, []).
        """
        node = self.nodes.get(name)
        if not node:
            # Unconstrained utility step (e.g. weather lookup)
            return True, []

        unmet = [dep for dep in sorted(node.dependencies) if dep in self.nodes and not self.nodes[dep].completed]
        return len(unmet) == 0, unmet

    def mark_completed(self, name: str, result: Optional[str] = None):
        """Marks a workflow step as completed."""
        if name in self.nodes:
            self.nodes[name].completed = True
            self.nodes[name].result = result

    def get_ready_steps(self) -> Set[str]:
        """Returns all steps whose dependencies have been satisfied but are not yet completed."""
        ready = set()
        for name, node in self.nodes.items():
            if not node.completed:
                can_run, _ = self.can_execute(name)
                if can_run:
                    ready.add(name)
        return ready

    def reset(self):
        """Resets all nodes to uncompleted state."""
        for node in self.nodes.values():
            node.completed = False
            node.result = None

    def render_ascii(self) -> str:
        """Renders an ASCII visualization of the DAG status and dependencies."""
        lines = ["\n📊 Workflow DAG Status:"]
        for name, node in self.nodes.items():
            status_icon = "✅ COMPLETED" if node.completed else "⏳ PENDING"
            if node.dependencies:
                dep_str = f"requires: {', '.join(sorted(node.dependencies))}"
            else:
                dep_str = "entry step"
            lines.append(f"  • [{status_icon}] {name:18} ({dep_str})")
        return "\n".join(lines)


def create_flight_booking_dag() -> WorkflowDAG:
    """
    Standard flight booking DAG:
    1. search_flights (Entry)
    2. get_baggage_policy (Depends on search_flights)
    3. calculator (Depends on search_flights)
    4. book_flight (Depends on search_flights and calculator)
    """
    dag = WorkflowDAG()
    dag.add_step(
        name="search_flights",
        dependencies=[],
        description="Search available flights and schedules between cities.",
    )
    dag.add_step(
        name="get_baggage_policy",
        dependencies=["search_flights"],
        description="Check airline baggage rules and fees for searched flight.",
    )
    dag.add_step(
        name="calculator",
        dependencies=["search_flights"],
        description="Calculate total price including tickets, baggage, and passengers.",
    )
    dag.add_step(
        name="book_flight",
        dependencies=["search_flights", "calculator"],
        description="Book ticket after pricing is verified and human confirms.",
    )
    return dag
