#!/usr/bin/env python3
"""
LangChain 1.0 Demo App

A comprehensive demonstration of LangChain 1.0's powerful features:
- LCEL (LangChain Expression Language) for simple chain composition
- Function/Tool calling with automatic model binding
- LangGraph for stateful, multi-actor workflows

Run this script to see all demos, or run individual demo files in the demos/ directory.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_environment():
    """Check if the environment is properly configured"""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "="*60)
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("="*60)
        print("\nPlease set up your environment:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to the .env file")
        print("3. Run this script again")
        print("\nExample .env file:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("  OPENAI_MODEL=gpt-4o-mini")
        print()
        return False
    return True


def print_header():
    """Print the demo app header"""
    print("\n" + "="*70)
    print(" "*15 + "🚀 LangChain 1.0 Demo App 🚀")
    print("="*70)
    print("""
This demo showcases the simplicity and power of LangChain 1.0!

📚 What you'll see:
   1. LCEL - Simple chain composition with the | operator
   2. Tools - Automatic function calling and execution
   3. LangGraph - Stateful workflows and agent graphs

Each demo is self-contained and shows real working examples.
""")
    print("="*70 + "\n")


def print_menu():
    """Print the interactive menu"""
    print("\n" + "─"*60)
    print("Select a demo to run:")
    print("─"*60)
    print("  1. LCEL Basics - Chain composition, streaming, and batch processing")
    print("  2. Function/Tool Calling - Automatic tool binding and execution")
    print("  3. LangGraph - Stateful workflows and conditional routing")
    print("  4. Run ALL demos (recommended for first-time viewers)")
    print("  0. Exit")
    print("─"*60)


def run_lcel_demo():
    """Run the LCEL demo"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_01_basic_lcel", "demos/01_basic_lcel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_tools_demo():
    """Run the tools demo"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_02_tools", "demos/02_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_langgraph_demo():
    """Run the LangGraph demo"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_03_langgraph", "demos/03_langgraph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def run_all_demos():
    """Run all demos in sequence"""
    print("\n" + "="*70)
    print(" "*20 + "Running All Demos")
    print("="*70)

    demos = [
        ("LCEL Basics", run_lcel_demo),
        ("Function/Tool Calling", run_tools_demo),
        ("LangGraph Workflows", run_langgraph_demo)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'='*70}")
        print(f"  Demo {i}/{len(demos)}: {name}")
        print('='*70)

        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\n❌ Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            response = input("\nContinue to next demo? (y/n): ")
            if response.lower() != 'y':
                break

        if i < len(demos):
            input("\nPress Enter to continue to the next demo...")

    print("\n" + "="*70)
    print(" "*22 + "All Demos Complete!")
    print("="*70 + "\n")


def run_interactive():
    """Run the interactive demo selector"""
    while True:
        print_menu()

        try:
            choice = input("\nEnter your choice (0-4): ").strip()

            if choice == "0":
                print("\n👋 Thanks for exploring LangChain 1.0!\n")
                break

            elif choice == "1":
                run_lcel_demo()

            elif choice == "2":
                run_tools_demo()

            elif choice == "3":
                run_langgraph_demo()

            elif choice == "4":
                run_all_demos()

            else:
                print("\n❌ Invalid choice. Please enter a number between 0 and 4.")

            if choice in ["1", "2", "3"]:
                input("\nPress Enter to return to menu...")

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for exploring LangChain 1.0!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point for the demo app"""
    # Check environment
    if not check_environment():
        sys.exit(1)

    # Print header
    print_header()

    # Check for command-line arguments
    if len(sys.argv) > 1:
        demo_choice = sys.argv[1].lower()

        if demo_choice in ["lcel", "1"]:
            run_lcel_demo()
        elif demo_choice in ["tools", "2"]:
            run_tools_demo()
        elif demo_choice in ["langgraph", "graph", "3"]:
            run_langgraph_demo()
        elif demo_choice in ["all", "4"]:
            run_all_demos()
        else:
            print(f"❌ Unknown demo: {demo_choice}")
            print("\nUsage: python demo_app.py [lcel|tools|langgraph|all]")
            print("   Or run without arguments for interactive mode")
            sys.exit(1)
    else:
        # Run interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
