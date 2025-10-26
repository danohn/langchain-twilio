"""
Demo 1: LCEL (LangChain Expression Language) Basics

This demo showcases the simplicity and power of LCEL in LangChain 1.0:
- Chain composition using the | operator
- Prompt templates
- Output parsing
- Streaming responses
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def demo_basic_chain():
    """Demonstrates basic chain composition with LCEL"""
    print("\n" + "="*60)
    print("Demo 1a: Basic Chain with LCEL")
    print("="*60)

    # Initialize the model
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts in simple terms."),
        ("human", "Explain {topic} in 2-3 sentences.")
    ])

    # Create an output parser
    output_parser = StrOutputParser()

    # Compose the chain using the | operator - This is the key feature of LCEL!
    chain = prompt | model | output_parser

    # Invoke the chain
    print("\nAsking: 'Explain quantum computing in 2-3 sentences'\n")
    result = chain.invoke({"topic": "quantum computing"})
    print(f"Response: {result}")


def demo_streaming():
    """Demonstrates streaming responses with LCEL"""
    print("\n" + "="*60)
    print("Demo 1b: Streaming with LCEL")
    print("="*60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative storyteller."),
        ("human", "Write a very short story (3 sentences) about {subject}.")
    ])

    # Same chain composition, but we'll use stream() instead of invoke()
    chain = prompt | model | StrOutputParser()

    print("\nAsking: 'Write a very short story about a robot learning to paint'\n")
    print("Response (streaming): ", end="", flush=True)

    # Stream the response
    for chunk in chain.stream({"subject": "a robot learning to paint"}):
        print(chunk, end="", flush=True)
    print("\n")


def demo_multi_step_chain():
    """Demonstrates a multi-step chain with LCEL"""
    print("\n" + "="*60)
    print("Demo 1c: Multi-Step Chain")
    print("="*60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # First chain: Generate a topic
    topic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You suggest interesting topics."),
        ("human", "Suggest one interesting topic about {category}. Reply with just the topic, nothing else.")
    ])
    topic_chain = topic_prompt | model | StrOutputParser()

    # Second chain: Explain the topic
    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You explain topics simply."),
        ("human", "Explain this topic in one sentence: {topic}")
    ])
    explain_chain = explain_prompt | model | StrOutputParser()

    print("\nStep 1: Generating a topic about 'space exploration'...")
    topic = topic_chain.invoke({"category": "space exploration"})
    print(f"Generated topic: {topic}")

    print("\nStep 2: Explaining the topic...")
    explanation = explain_chain.invoke({"topic": topic})
    print(f"Explanation: {explanation}")


def demo_batch_processing():
    """Demonstrates batch processing with LCEL"""
    print("\n" + "="*60)
    print("Demo 1d: Batch Processing")
    print("="*60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You provide brief, one-sentence definitions."),
        ("human", "Define {term} in one sentence.")
    ])

    chain = prompt | model | StrOutputParser()

    # Batch process multiple inputs
    terms = ["AI", "Machine Learning", "Neural Networks"]
    print(f"\nDefining multiple terms: {terms}\n")

    results = chain.batch([{"term": term} for term in terms])

    for term, definition in zip(terms, results):
        print(f"• {term}: {definition}")


def main():
    """Run all LCEL demos"""
    print("\n" + "="*60)
    print("LangChain 1.0 - LCEL Basics Demo")
    print("="*60)
    print("\nLCEL (LangChain Expression Language) makes building chains")
    print("simple and intuitive using the | operator!")

    try:
        demo_basic_chain()
        demo_streaming()
        demo_multi_step_chain()
        demo_batch_processing()

        print("\n" + "="*60)
        print("All LCEL demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")


if __name__ == "__main__":
    main()
