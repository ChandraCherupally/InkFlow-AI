# Beyond Linear Chains: The Shift to Loop and Graph Engineering

We began our LLM journey by building chains. We linked Prompt A to Prompt B, hoping a predictable, sequential pipeline would yield a perfect result. This approach, however, reflects a fundamental misunderstanding of what makes AI powerful. Real-world tasks are messy, unpredictable, and rarely solved in a single shot.



![Comparison of linear LLM chains vs. resilient loop and graph architectures.](/images/linear_vs_loop_graph_engineering.png)
*Figure 1: The structural shift from fragile linear LLM pipelines to robust, self-correcting state machines.*



> ⚠️ **Common Mistake:** When an LLM makes a mistake in step two of a ten-step linear chain, the entire pipeline collapses with no way to recover. Linear LLM chains lack situational awareness and cannot backtrack, much like a factory conveyor belt blindly welding over a crooked part.

> ✅ **Best Practice:** We are in the midst of a paradigm shift: moving from writing rigid, static prompts to engineering dynamic, recursive systems that analyze, critique, and refine their own outputs.

## Loop Engineering: Building Self-Correcting Engines

This new paradigm is called **Loop Engineering**. It's an architectural pattern where an agent continuously executes, evaluates, and refines its own output in an autonomous cycle until a strictly defined goal is met.

> ✅ **Best Practice:** Instead of a developer hardcoding every transition, the system operates inside a state loop governed by its own logic.

Think of it like a smart thermostat. You don't just tell a heater to run for five minutes and hope for the best. You set a target temperature, and the thermostat continuously measures the room, activates the heat, evaluates the change, and stops when the target state is reached. This is the essence of a feedback loop.

In a loop-engineered system, this cycle is driven by three technical pillars:

*   **The Generator:** The agentic core that attempts to solve the user's prompt.
*   **The Evaluator:** A validation step—often code-based execution or a secondary LLM judge—that tests the generator's output.
*   **The Feedback Loop:** The mechanism that feeds execution errors or critique back to the generator, prompting a recursive rewrite.



![Feedback loop architecture showing Generator, Evaluator, and Feedback mechanisms.](/images/loop_engineering_feedback_cycle.png)
*Figure 2: The Loop Engineering pattern showing the self-correcting cycle driven by feedback.*



The following Python example shows a simple self-correcting loop. A generator attempts to write JSON, an evaluator checks for syntax errors, and the loop automatically feeds errors back to the generator until the output passes validation.

```python
import json

# Simulated LLM that "learns" from feedback
class MockLLMGenerator:
    def __init__(self):
        self.attempts = 0

    def generate_json_string(self, feedback: str = None) -> str:
        self.attempts += 1
        # On the first attempt, the LLM makes a common syntax error (trailing comma)
        if self.attempts == 1:
            return '{"project": "Loop Engineering", "status": "active",}'
        
        # On the second attempt, it uses the feedback to correct the output
        if "Trailing comma" in feedback:
            return '{"project": "Loop Engineering", "status": "active"}'
        
        return '{"error": "unknown"}'

def evaluate_json(json_str: str) -> tuple[bool, str]:
    """Validates the output and returns success status with feedback."""
    try:
        json.loads(json_str)
        return True, "Success! Valid JSON."
    except json.JSONDecodeError as e:
        # Return the exact parsing error back to the loop
        return False, f"JSONDecodeError: {str(e)}"

# --- The Loop Engineering Execution ---
generator = MockLLMGenerator()
max_iterations = 3
feedback = ""
success = False

print("Starting Self-Correcting Loop...\n")

for step in range(1, max_iterations + 1):
    print(f"--- Iteration {step} ---")
    
    # 1. Generate: The agent attempts the task, using prior feedback
    output = generator.generate_json_string(feedback=feedback)
    print(f"Generator Output: {output}")
    
    # 2. Evaluate: An independent process validates the output
    is_valid, feedback = evaluate_json(output)
    
    # 3. Loop or Exit: The system decides whether to retry or succeed
    if is_valid:
        print(f"\n[Verified] Output passed on step {step}!")
        success = True
        break
    else:
        print(f"[Failed] Feedback generated: '{feedback}'")
        print("Routing back to Generator...\n")

if not success:
    print("Loop terminated: Max iterations reached without success.")
```

## Graph Engineering: Imposing Order on Agentic Loops

> ⚠️ **Common Mistake:** While loops grant agents the power to self-correct, raw, unconstrained autonomy is a recipe for production disaster. If we let an LLM decide its own next steps without boundaries, we invite infinite loops, hallucinated tool calls, and runaway token bills.

> ✅ **Best Practice:** To deploy agents confidently, we must transition from open-ended autonomy to guided, predictable workflows. This is where **Graph Engineering** comes in.

Graph Engineering is the practice of structuring agentic behavior as a deterministic state machine.

> 💡 **Tip:** Instead of letting the model guess the software architecture at runtime, we enforce strict boundaries by defining the application's flow as a directed graph.

To visualize this, imagine an autonomous delivery vehicle. Placing it in an open field without a map might lead to the destination, but it will more likely get stuck in a ditch. Graph engineering is like laying down **train tracks** and **switching stations**. The train (our LLM) still controls its speed and makes local decisions at junctions, but it can never derail or wander off into unapproved territory.

An agentic graph relies on three core components:

*   **Nodes:** Isolated functions that perform specific actions, like calling an LLM, querying a database, or formatting a response.
*   **Edges:** The "tracks" that define the direct pathways from one node to another.
*   **Conditional Edges:** Decision-making routers that evaluate the application's current state to determine which node to execute next.

At the center of this architecture is the **Shared State**, a persistent data object passed between nodes. Nodes read from and write to this single source of truth, creating a clean, observable, and debuggable workflow.



![Graph engineering architecture with nodes, edges, and shared state.](/images/graph_engineering_state_machine.png)
*Figure 3: Graph Engineering workflow acting as a deterministic state machine built around a central Shared State.*



The following code demonstrates a simple state graph for a customer support workflow. It routes queries based on a shared state context, ensuring the system follows a predictable path.

```python
from typing import Dict, Any, Callable

# 1. Define the Shared State structure
class AgentState:
    def __init__(self, query: str):
        self.state: Dict[str, Any] = {
            "query": query,
            "category": None,
            "response": None,
            "steps_taken": 0
        }

# 2. Define the Nodes (Actions)
def classify_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[Node: Classify] Analyzing customer query...")
    # Simulate classification logic (could be an LLM call)
    query = state["query"].lower()
    if "refund" in query or "billing" in query:
        state["category"] = "billing"
    else:
        state["category"] = "general"
    state["steps_taken"] += 1
    return state

def billing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[Node: Billing] Routing to payment gateway API...")
    state["response"] = "Processed refund request successfully."
    state["steps_taken"] += 1
    return state

def general_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[Node: General] Consulting knowledge base...")
    state["response"] = "Here is the answer to your general inquiry."
    state["steps_taken"] += 1
    return state

# 3. Define the Router (Conditional Edge)
def route_next_step(state: Dict[str, Any]) -> str:
    # Explicit boundary: Prevent infinite execution loops
    if state["steps_taken"] > 3:
        return "END"
    
    if state["category"] == "billing":
        return "billing_node"
    return "general_node"

# 4. The Graph Orchestrator
class SimpleGraphEngine:
    def __init__(self):
        self.nodes: Dict[str, Callable] = {}
        
    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func

    def run(self, initial_state: AgentState) -> Dict[str, Any]:
        current_state = initial_state.state
        
        # Start at the entry point node
        current_state = self.nodes["classify"](current_state)
        
        # Route dynamically based on the state evaluation
        next_step_name = route_next_step(current_state)
        
        if next_step_name != "END" and next_step_name in self.nodes:
            next_node = self.nodes[next_step_name]
            current_state = next_node(current_state)
            
        return current_state

# --- Execution ---
engine = SimpleGraphEngine()
engine.add_node("classify", classify_query_node)
engine.add_node("billing_node", billing_node)
engine.add_node("general_node", general_node)

user_state = AgentState("I need a refund on my last billing cycle.")
final_state = engine.run(user_state)
print(f"\nFinal State Response: {final_state['response']}")
```

### Blueprint: Implementing a Self-Correcting Node

> 💡 **Tip:** The true power of this architecture emerges when we combine Loop and Graph Engineering. We can implement a self-correcting loop *inside* a single node of our graph. This creates a resilient, atomic step that guarantees its output is valid before passing control to the next part of the system.

At the core of this pattern is the separation of creative and critical tasks.

> ⚠️ **Common Mistake:** If the same LLM instance generates code and then verifies it, it will often overlook its own subtle bugs due to confirmation bias.

> ✅ **Best Practice:** To solve this, we split the node into two distinct roles:
> *   **The Generator:** Focuses entirely on drafting the output.
> *   **The Evaluator:** An independent process optimized for finding flaws.

Think of it like a restaurant kitchen. The line cook (Generator) prepares the dish, but it can't be served directly. Instead, the head chef (Evaluator) inspects it at the pass, sending it back with specific feedback if the seasoning is off. This quality control gate ensures only high-quality output leaves the kitchen.

The diagram below shows how a self-correcting loop operates within a graph. The router sends the state back to the generator if evaluation fails, only allowing validated output to exit the cycle.



![Detailed architectural diagram of a self-correcting node containing a Generator-Evaluator loop.](/images/self_correcting_node_blueprint.png)
*Figure 4: Deep dive blueprint of an atomic, self-correcting node combining loop and graph concepts.*



```text
     +--------------------+
     |   Generator Node   | <-----------------------+
     +--------+-----------+                         |
              |                                     |
              v                                     | (Feedback on Failure)
     +--------------------+                         |
     |   Evaluator Node   | -- (Router Logic) ------+
     +--------+-----------+
              |
              | (On Success or Max Retries)
              v
       +------------------+
       |   Next Node or   |
       |       Exit       |
       +------------------+
```

The following implementation demonstrates this pattern. We define a graph state, a Generator, an Evaluator, and a router that decides whether to retry or proceed.

```python
import json
from typing import Dict, Any

# Simple simulated LLM client
def mock_llm_call(prompt: str) -> str:
    # Simulates an LLM returning flawed code first, then correcting it based on feedback
    if "Fix this error" in prompt:
        return '{"code": "def add(a, b): return a + b"}'
    return '{"code": "def add(a, b): return a  b"}'

def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates code based on the current prompt and past feedback."""
    print(f"\n[Generator] Running attempt {state['attempts'] + 1}...")
    
    prompt = "Write a Python function to add two numbers."
    if state["feedback"]:
        prompt += f" Fix this error from your previous attempt: {state['feedback']}"
        
    raw_response = mock_llm_call(prompt)
    response_data = json.loads(raw_response)
    
    state["code"] = response_data["code"]
    state["attempts"] += 1
    return state

def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the code for structural errors."""
    print("[Evaluator] Checking generated code...")
    code = state["code"]
    
    # Deterministic check: Does the code contain a '+' operator?
    if "+" not in code:
        state["is_valid"] = False
        state["feedback"] = "The operator '+' is missing from the addition logic."
    else:
        state["is_valid"] = True
        state["feedback"] = "Code successfully passed all checks."
        
    return state

def route_next_step(state: Dict[str, Any]) -> str:
    """Determines whether to loop back or exit."""
    if state["is_valid"]:
        print("[Router] Validation passed. Routing to: SUCCESS")
        return "success"
    
    if state["attempts"] >= state["max_attempts"]:
        print("[Router] Max attempts reached. Routing to: FAILURE")
        return "fail"
    
    print("[Router] Validation failed. Routing to: GENERATOR for retry.")
    return "retry"

# --- Executing the Graph Workflow ---
graph_state = {
    "code": "",
    "attempts": 0,
    "max_attempts": 3,
    "is_valid": False,
    "feedback": None
}

while True:
    graph_state = generator_node(graph_state)
    graph_state = evaluator_node(graph_state)
    
    next_step = route_next_step(graph_state)
    if next_step in ["success", "fail"]:
        break

print(f"\nFinal State: {json.dumps(graph_state, indent=2)}")
```

## Production Safeguards: Taming Autonomous Agents

> ⚠️ **Common Mistake:** When you give code the agency to decide its own control flow, you must prepare for the worst. A minor parsing error or ambiguous prompt can cause an agent to spin in an endless circle, draining your API budget before alerts even trigger.

> 🚀 **Production Tip:** The golden rule of Loop Engineering is to **never deploy a loop without a hard-coded escape hatch.**

Think of it like a robotic vacuum. If it gets stuck in a closet, it doesn't bump against the walls forever. It has built-in overrides: a collision counter, a runtime limit, and an error alarm. We must build these same physical bumpers into our software.

### 1. Guarding the Gate: Cycle Caps and Budgets

> 🚀 **Production Tip:** Enforce limits at the orchestration layer by building hard caps directly into your graph's state:
> *   **Max Iterations:** A strict counter that throws an exception when breached.
> *   **Token Budgets:** Track cumulative token consumption per node and abort the run if it exceeds a set dollar amount.
> *   **Timeouts:** A maximum wall-clock execution time for the entire graph run.

### 2. Schema Enforcement: Preventing State Corruption

> ⚠️ **Common Mistake:** In a looping graph, data is passed from one node to the next. If Node A returns a mutated payload that Node B doesn't expect, the graph can crash.

> ✅ **Best Practice:** To prevent this, use structured schema validation with libraries like **Pydantic** at every node transition.

The schema below enforces strict types and value ranges, ensuring that an LLM cannot pass a hallucinated `confidence_score` of `1.5` downstream.

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class AgentOutputSchema(BaseModel):
    """Enforces strict structural boundaries for loop outputs."""
    thought_process: str
    suggested_actions: List[str]
    confidence_score: float

    @field_validator("confidence_score")
    def validate_score(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return value
```

> 💡 **Tip:** If an LLM returns a payload that violates this schema, Pydantic raises an error immediately. Instead of looping on bad data, the system can gracefully route to a healing node or fail loudly.

### 3. Observability: Tracking Agent Trajectories

> 🚀 **Production Tip:** When an agent loops five times before failing, you need to know *when* and *in which cycle* the state went off-course. This requires explicit **trajectory tracing**. Log each state transition as a unique event in a monitoring tool like LangSmith or OpenInference.

> 💡 **Tip:** By capturing immutable snapshots of the graph state at each iteration, you gain the ability to "time travel" through the agent's execution history. If a failure occurs on step three, you can load the exact state from step two into your local debugger to instantly replicate and diagnose the issue.

## The Future is Composable and Resilient

> ✅ **Best Practice:** We are witnessing a fundamental shift from "prompt engineering" to "systems engineering." The era of relying on fragile, single-shot instructions is fading. In its place, Loop and Graph Engineering provide robust, system-level design patterns for building intelligent applications.

> ✅ **Best Practice:** By modeling workflows as directed state graphs, we strike a perfect balance between the creative reasoning of LLMs and the deterministic constraints of traditional software. The graph provides the ultimate guardrail, ensuring that while the LLM has freedom to reason, the system architecture guarantees predictable paths and reliable error recovery.

> 💡 **Tip:** You don't need to build a complex multi-agent network overnight to benefit from this paradigm. Start small. Identify a single, high-failure step in your current pipeline—like generating valid JSON—and wrap just that one step in a self-correcting **micro-loop**.

```python
import json
from typing import Dict, Any, Callable

# A flaky function that simulates an LLM call
def mock_llm_json_generator(attempt: int) -> str:
    if attempt == 1:
        # First attempt has two common JSON errors
        return "{'name': 'Agent', 'status': incomplete}"
    return '{"name": "Agent", "status": "active"}'

def run_micro_loop(generator: Callable[[int], str], max_attempts: int = 3) -> Dict[str, Any]:
    """Wraps a flaky step in a deterministic micro-loop to ensure valid output."""
    for attempt in range(1, max_attempts + 1):
        raw_output = generator(attempt)
        try:
            # Attempt to parse and validate the output
            data = json.loads(raw_output)
            print(f"[Success] Validated data on attempt {attempt}!")
            return data
        except json.JSONDecodeError as e:
            # Instead of crashing, capture the error and loop back for a retry
            print(f"[Attempt {attempt} Failed] Error: {e}. Retrying...")
    
    raise ValueError("System failed to generate valid JSON within the attempt limit.")

# Run the micro-loop to see self-correction in action
result = run_micro_loop(mock_llm_json_generator)
print(f"Final Structured Output: {result}")
```

> ✅ **Best Practice:** This simple micro-loop isolates the fragile LLM step from the rest of the application. By trapping errors in a retry block, we guarantee the wider system only receives reliable, structured data. By shifting your mindset from writing prompts to engineering systems, you stop hoping for reliability and start guaranteeing it.

## Key Takeaways
*   Linear LLM chains are fragile and prone to collapse upon error.
*   Loop Engineering enables self-correction through a Generator, Evaluator, and Feedback Loop.
*   Graph Engineering structures agent behavior into predictable state machines using Nodes, Edges, and Shared State.
*   Combining loops within graph nodes creates resilient, atomic self-correcting steps.
*   Production systems require hard-coded escape hatches, schema validation, and robust observability for autonomous agents.

---

## SEO Keywords
- LLM Engineering
- Loop Engineering
- Graph Engineering
- Self-Correcting AI
- Agentic Systems