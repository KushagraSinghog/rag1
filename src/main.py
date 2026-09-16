import argparse
import sys
from pathlib import Path

# Ensure src directory is in path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chat import chat


def run_cli():
    print("=" * 60)
    print("Technical Book RAG Agent - Interactive CLI Mode")
    print("Type 'exit' or 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            results = chat(query)

            print("\nAssistant: ")
            print(results.get("answer", ""))

            if results.get("sources"):
                print(f"\nSources (Pages): {results['sources']}")

            if results.get("latency_seconds"):
                print(f"Latency: {results['latency_seconds']}s")

            print("\n" + "-" * 40 + "\n")
        except KeyboardInterrupt:
            print("\nExiting CLI mode.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    import uvicorn
    print(f"Starting Technical Book RAG FastAPI Server...")
    print(f"Interactive Web UI: http://{host}:{port}/ui")
    print(f"OpenAPI Docs:       http://{host}:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=reload, app_dir=str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(description="Technical Book RAG Agent & FastAPI Server")
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI interface instead of FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for FastAPI server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for FastAPI server")

    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()