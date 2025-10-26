"""
Demo 2: Function/Tool Calling

This demo showcases LangChain 1.0's improved tool/function calling:
- Defining custom tools with the @tool decorator
- Binding tools to models
- Automatic function execution
- Tool chains with LCEL
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Load environment variables
load_dotenv()


# Define custom tools using the @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5")
    """
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """Counts the number of words in a given text.

    Args:
        text: The text to count words in
    """
    word_count = len(text.split())
    return f"The text contains {word_count} words."


@tool
def string_reverser(text: str) -> str:
    """Reverses a given string.

    Args:
        text: The text to reverse
    """
    reversed_text = text[::-1]
    return f"Reversed text: {reversed_text}"


@tool
def temperature_converter(temperature: float, from_unit: str, to_unit: str) -> str:
    """Converts temperature between Celsius and Fahrenheit.

    Args:
        temperature: The temperature value to convert
        from_unit: The source unit ('C' for Celsius or 'F' for Fahrenheit)
        to_unit: The target unit ('C' for Celsius or 'F' for Fahrenheit)
    """
    try:
        from_unit = from_unit.upper()
        to_unit = to_unit.upper()

        if from_unit == to_unit:
            return f"{temperature}°{from_unit}"

        if from_unit == 'C' and to_unit == 'F':
            result = (temperature * 9/5) + 32
            return f"{temperature}°C = {result:.2f}°F"
        elif from_unit == 'F' and to_unit == 'C':
            result = (temperature - 32) * 5/9
            return f"{temperature}°F = {result:.2f}°C"
        else:
            return "Invalid units. Use 'C' for Celsius or 'F' for Fahrenheit."
    except Exception as e:
        return f"Error converting temperature: {str(e)}"


def demo_basic_tool_calling():
    """Demonstrates basic tool binding and calling"""
    print("\n" + "="*60)
    print("Demo 2a: Basic Tool Calling")
    print("="*60)

    # Initialize model
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Define tools
    tools = [calculator, word_counter, string_reverser]

    # Bind tools to the model - This is a key LangChain 1.0 feature!
    model_with_tools = model.bind_tools(tools)

    # Invoke with a query that requires tool use
    print("\nQuery: 'What is 42 multiplied by 17?'\n")

    messages = [HumanMessage(content="What is 42 multiplied by 17?")]
    response = model_with_tools.invoke(messages)

    print(f"Model's response: {response.content}")

    # Check if the model called any tools
    if response.tool_calls:
        print(f"\nTools called: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  • Tool: {tool_call['name']}")
            print(f"    Arguments: {tool_call['args']}")


def demo_multiple_tools():
    """Demonstrates using multiple tools in a single query"""
    print("\n" + "="*60)
    print("Demo 2b: Multiple Tools")
    print("="*60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    tools = [calculator, word_counter, temperature_converter]
    model_with_tools = model.bind_tools(tools)

    # Query that might use multiple tools
    query = "Calculate 15 + 27, and also convert 25 Celsius to Fahrenheit"
    print(f"\nQuery: '{query}'\n")

    messages = [HumanMessage(content=query)]
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"Model decided to use {len(response.tool_calls)} tool(s):\n")

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"Executing {tool_name} with args: {tool_args}")

            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    print(f"Result: {result}\n")
                    break
    else:
        print(f"Model response: {response.content}")


def demo_tool_chain():
    """Demonstrates creating a chain with tools"""
    print("\n" + "="*60)
    print("Demo 2c: Tool Chain with LCEL")
    print("="*60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    tools = [calculator, word_counter, string_reverser, temperature_converter]
    model_with_tools = model.bind_tools(tools)

    # Create a prompt that encourages tool use
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools. "
                   "Use the appropriate tools to answer the user's questions accurately."),
        ("human", "{query}")
    ])

    # Create a chain
    chain = prompt | model_with_tools

    queries = [
        "How many words are in this sentence: 'LangChain 1.0 is amazing'?",
        "What's 100 divided by 4?",
        "Reverse this text: 'hello world'"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        response = chain.invoke({"query": query})

        if response.tool_calls:
            print(f"Tool(s) called: {[tc['name'] for tc in response.tool_calls]}")

            # Execute the first tool call as an example
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                for tool in tools:
                    if tool.name == tool_call['name']:
                        result = tool.invoke(tool_call['args'])
                        print(f"Result: {result}")
                        break
        else:
            print(f"Response: {response.content}")


def demo_tool_descriptions():
    """Shows how tool descriptions help the model choose the right tool"""
    print("\n" + "="*60)
    print("Demo 2d: Tool Descriptions Matter")
    print("="*60)

    print("\nOur tools have clear descriptions that help the model choose correctly:")

    tools = [calculator, word_counter, string_reverser, temperature_converter]

    for tool in tools:
        print(f"\n• {tool.name}")
        print(f"  Description: {tool.description}")


def main():
    """Run all tool calling demos"""
    print("\n" + "="*60)
    print("LangChain 1.0 - Function/Tool Calling Demo")
    print("="*60)
    print("\nLangChain 1.0 makes tool calling simple and intuitive!")
    print("Models can automatically choose and execute the right tools.")

    try:
        demo_basic_tool_calling()
        demo_multiple_tools()
        demo_tool_chain()
        demo_tool_descriptions()

        print("\n" + "="*60)
        print("All tool calling demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")


if __name__ == "__main__":
    main()
