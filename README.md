# LangChain 1.0 Demo App

A comprehensive demo showcasing the simplicity and powerful new features of LangChain 1.0.

## Features Demonstrated

### 1. **LCEL (LangChain Expression Language)**
- Simple, intuitive chain composition with the `|` operator
- Easy prompt templating and output parsing
- Streamlined conversation flows

### 2. **Function/Tool Calling**
- Native tool integration with LLMs
- Automatic function binding and execution
- Real-world utility functions (calculator, weather, etc.)

### 3. **LangGraph - Stateful Workflows**
- Multi-step agent workflows with state management
- Conditional branching and routing
- Cyclic graph support for complex agent behaviors

### 4. **Streaming Support**
- Real-time token streaming for better UX
- Async streaming capabilities

## Project Structure

```
.
├── README.md
├── requirements.txt
├── demo_app.py              # Main demo runner
├── demos/
│   ├── 01_basic_lcel.py     # LCEL basics
│   ├── 02_tools.py          # Function/tool calling
│   └── 03_langgraph.py      # Stateful workflows with LangGraph
└── .env.example             # Environment variables template
```

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Usage

Run all demos:
```bash
python demo_app.py
```

Run individual demos:
```bash
python demos/01_basic_lcel.py
python demos/02_tools.py
python demos/03_langgraph.py
```

## What's New in LangChain 1.0?

### Simplified Chain Building with LCEL
LangChain 1.0 introduces a cleaner, more intuitive way to build chains:

```python
chain = prompt | model | output_parser
result = chain.invoke({"topic": "AI"})
```

### Enhanced Tool/Function Calling
Tools are now first-class citizens with automatic binding:

```python
tools = [calculator, search]
model_with_tools = model.bind_tools(tools)
```

### LangGraph for Complex Workflows
Build stateful, multi-actor applications with ease:

```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_edge("agent", "tools")
```

## Requirements

- Python 3.9+
- OpenAI API key (or other LLM provider)
- Internet connection for API calls

## License

MIT License - Feel free to use this demo for learning and development!
