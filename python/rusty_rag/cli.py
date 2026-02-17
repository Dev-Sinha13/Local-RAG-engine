"""RustyRAG CLI — Chat with your local PDF documents using RAG."""

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """RustyRAG — Chat with your local PDF documents using RAG.

    Powered by Ollama (local LLM) and Qdrant (vector database).
    All data stays on your machine. Zero data leakage.

    Performance-critical operations (PDF parsing, text chunking) run in Rust.
    High-level orchestration (embeddings, LLM, vector search) runs in Python.
    """
    load_dotenv()


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
def ingest(file_path: str):
    """Ingest a PDF file into the knowledge base.

    Extracts text from the PDF, splits it into semantic chunks,
    generates embeddings, and stores everything in Qdrant.
    """
    from .rag import ingest as do_ingest

    try:
        do_ingest(file_path)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        raise SystemExit(1)


@main.command()
@click.argument("question")
def query(question: str):
    """Query the knowledge base with a question.

    Searches for relevant chunks in the vector database,
    then uses the LLM to generate an answer based on the context.
    """
    from .rag import query as do_query

    try:
        response = do_query(question)
        console.print()
        console.print(Panel(response, title="📝 Answer", border_style="green"))
        console.print()
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
