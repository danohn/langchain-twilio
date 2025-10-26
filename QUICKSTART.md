# Quick Start Guide

Get up and running with the LangChain 1.0 demo in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

## Setup Steps

### 1. Clone and Navigate

```bash
cd langchain-twilio
```

### 2. Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# On macOS/Linux: nano .env
# On Windows: notepad .env
```

Add your API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

### 5. Run the Demo!

```bash
# Interactive mode (recommended for first time)
python demo_app.py

# Or run all demos at once
python demo_app.py all

# Or run individual demos
python demo_app.py lcel
python demo_app.py tools
python demo_app.py langgraph
```

## What You'll See

### Demo 1: LCEL (LangChain Expression Language)
Simple, elegant chain composition:
```python
chain = prompt | model | output_parser
result = chain.invoke({"topic": "AI"})
```

### Demo 2: Function/Tool Calling
Automatic tool binding and execution:
```python
@tool
def calculator(expression: str) -> str:
    """A simple calculator"""
    return str(eval(expression))

model_with_tools = model.bind_tools([calculator])
```

### Demo 3: LangGraph
Stateful workflows with conditional routing:
```python
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_conditional_edges("agent", should_continue)
app = workflow.compile()
```

## Troubleshooting

### "No module named 'langchain'"
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt` again

### "OPENAI_API_KEY not found"
- Check that you created the `.env` file
- Verify your API key is correctly pasted (no extra spaces)
- Make sure the file is named `.env` exactly (not `.env.txt`)

### API Rate Limits
- The demo uses `gpt-4o-mini` by default (cheaper and faster)
- You can change the model in `.env` if needed

## Next Steps

- Explore the code in `demos/` directory
- Modify the examples to experiment
- Check out the [LangChain documentation](https://python.langchain.com/)
- Read about [LangGraph](https://langchain-ai.github.io/langgraph/)

## Getting Help

- LangChain Docs: https://python.langchain.com/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- Discord: https://discord.gg/langchain

Enjoy exploring LangChain 1.0! 🚀
