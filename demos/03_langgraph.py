"""
Demo 3: LangGraph - Stateful Workflows

This demo showcases LangGraph, one of the most powerful features in LangChain 1.0:
- Building stateful, multi-step workflows
- Creating agent graphs with conditional routing
- Managing state across multiple nodes
- Cyclic graph execution
"""

import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# Load environment variables
load_dotenv()


# Define tools for the agent
@tool
def search_knowledge_base(query: str) -> str:
    """Searches a knowledge base for information.

    Args:
        query: The search query
    """
    # Simulated knowledge base
    knowledge = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "lcel": "LCEL (LangChain Expression Language) allows you to easily compose chains using the | operator."
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Found: {value}"

    return "No information found in the knowledge base."


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Define the state for our graph
class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str


def demo_simple_graph():
    """Demonstrates a simple LangGraph workflow"""
    print("\n" + "="*60)
    print("Demo 3a: Simple LangGraph Workflow")
    print("="*60)

    # Define a simple state
    class SimpleState(TypedDict):
        messages: Sequence[str]
        count: int

    # Define nodes
    def start_node(state: SimpleState) -> SimpleState:
        print("  → Executing: start_node")
        return {
            "messages": state["messages"] + ["Started workflow"],
            "count": 1
        }

    def process_node(state: SimpleState) -> SimpleState:
        print("  → Executing: process_node")
        return {
            "messages": state["messages"] + ["Processing data"],
            "count": state["count"] + 1
        }

    def end_node(state: SimpleState) -> SimpleState:
        print("  → Executing: end_node")
        return {
            "messages": state["messages"] + ["Workflow complete"],
            "count": state["count"] + 1
        }

    # Create the graph
    workflow = StateGraph(SimpleState)

    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("process", process_node)
    workflow.add_node("end", end_node)

    # Add edges (define the flow)
    workflow.set_entry_point("start")
    workflow.add_edge("start", "process")
    workflow.add_edge("process", "end")
    workflow.add_edge("end", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    print("\nRunning workflow...\n")
    result = app.invoke({"messages": [], "count": 0})

    print("\nWorkflow Results:")
    print(f"  Steps executed: {result['count']}")
    print(f"  Messages: {result['messages']}")


def demo_conditional_routing():
    """Demonstrates conditional routing in LangGraph"""
    print("\n" + "="*60)
    print("Demo 3b: Conditional Routing")
    print("="*60)

    # Define state
    class RouterState(TypedDict):
        number: int
        path_taken: str
        result: str

    # Define nodes
    def check_number(state: RouterState) -> RouterState:
        print(f"  → Checking number: {state['number']}")
        return state

    def positive_path(state: RouterState) -> RouterState:
        print("  → Taking positive path")
        return {
            "number": state["number"],
            "path_taken": "positive",
            "result": f"{state['number']} is positive"
        }

    def negative_path(state: RouterState) -> RouterState:
        print("  → Taking negative path")
        return {
            "number": state["number"],
            "path_taken": "negative",
            "result": f"{state['number']} is negative"
        }

    def zero_path(state: RouterState) -> RouterState:
        print("  → Taking zero path")
        return {
            "number": state["number"],
            "path_taken": "zero",
            "result": "Number is zero"
        }

    # Define conditional routing logic
    def route_based_on_number(state: RouterState) -> str:
        """Determines which path to take based on the number"""
        if state["number"] > 0:
            return "positive"
        elif state["number"] < 0:
            return "negative"
        else:
            return "zero"

    # Create the graph
    workflow = StateGraph(RouterState)

    # Add nodes
    workflow.add_node("check", check_number)
    workflow.add_node("positive", positive_path)
    workflow.add_node("negative", negative_path)
    workflow.add_node("zero", zero_path)

    # Set entry point
    workflow.set_entry_point("check")

    # Add conditional edges
    workflow.add_conditional_edges(
        "check",
        route_based_on_number,
        {
            "positive": "positive",
            "negative": "negative",
            "zero": "zero"
        }
    )

    # All paths lead to END
    workflow.add_edge("positive", END)
    workflow.add_edge("negative", END)
    workflow.add_edge("zero", END)

    # Compile the graph
    app = workflow.compile()

    # Test with different numbers
    test_numbers = [42, -10, 0]

    for num in test_numbers:
        print(f"\nTesting with number: {num}")
        result = app.invoke({"number": num, "path_taken": "", "result": ""})
        print(f"  Path taken: {result['path_taken']}")
        print(f"  Result: {result['result']}")


def demo_agent_with_tools():
    """Demonstrates a simple agent with tools using LangGraph"""
    print("\n" + "="*60)
    print("Demo 3c: Agent with Tools")
    print("="*60)

    # Initialize model with tools
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    tools = [search_knowledge_base, calculate]
    model_with_tools = model.bind_tools(tools)

    # Define agent state
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    # Define the agent node
    def call_model(state: AgentState) -> AgentState:
        """Call the model with the current state"""
        print("  → Agent thinking...")
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define routing logic
    def should_continue(state: AgentState) -> str:
        """Determine if we should use tools or end"""
        last_message = state["messages"][-1]

        # If there are tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"  → Agent wants to use {len(last_message.tool_calls)} tool(s)")
            return "tools"

        # Otherwise, we're done
        print("  → Agent is finished")
        return "end"

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)

    # Create tool node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # After using tools, go back to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    app = workflow.compile()

    # Test queries
    queries = [
        "What is LangGraph?",
        "Calculate 15 times 8"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)

        result = app.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # Get the final response
        final_message = result["messages"][-1]
        print(f"\nFinal Answer: {final_message.content}")


def demo_stateful_conversation():
    """Demonstrates maintaining state across multiple interactions"""
    print("\n" + "="*60)
    print("Demo 3d: Stateful Conversation")
    print("="*60)

    # Define state that tracks conversation history and context
    class ConversationState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        topic_count: int
        topics_discussed: Sequence[str]

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    def conversation_node(state: ConversationState) -> ConversationState:
        """Process a conversation turn"""
        response = model.invoke(state["messages"])

        # Simple topic extraction (in a real app, this would be more sophisticated)
        new_topics = state.get("topics_discussed", [])

        return {
            "messages": [response],
            "topic_count": len(state["messages"]) // 2,
            "topics_discussed": new_topics
        }

    # Create simple graph
    workflow = StateGraph(ConversationState)
    workflow.add_node("chat", conversation_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)

    app = workflow.compile()

    print("\nHaving a multi-turn conversation with state tracking...\n")

    # Simulate a conversation
    conversation_turns = [
        "Hello! My name is Alice.",
        "What's my name?",
        "Tell me a fun fact about space."
    ]

    state = {"messages": [], "topic_count": 0, "topics_discussed": []}

    for turn in conversation_turns:
        print(f"User: {turn}")

        state["messages"].append(HumanMessage(content=turn))
        result = app.invoke(state)

        # Update state with the full result
        state = result

        assistant_message = result["messages"][-1]
        print(f"Assistant: {assistant_message.content}")
        print(f"(Turn {result['topic_count']})\n")


def main():
    """Run all LangGraph demos"""
    print("\n" + "="*60)
    print("LangChain 1.0 - LangGraph Demo")
    print("="*60)
    print("\nLangGraph enables building stateful, multi-actor applications")
    print("with complex workflows, conditional routing, and tool usage!")

    try:
        demo_simple_graph()
        demo_conditional_routing()
        demo_agent_with_tools()
        demo_stateful_conversation()

        print("\n" + "="*60)
        print("All LangGraph demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")


if __name__ == "__main__":
    main()
